import logging
import os
import time
from datetime import datetime
from pathlib import Path
import sys
import requests
import json

# Append the workspace's root directory to the sys.path
sys.path.append(str(Path(__file__).parent.parent))

import colorlog
import mlflow
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from .config import RetrievalExpsConfig
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

from data_util import get_medical_data
from .evaluation import (
    load_test_queries,
    get_sentence_embedding,
    getMetrics
)
from .prompt import generatePrompt

from retrieval.inference import (preprocesado, retrieval, postprocesado)


CACHE_PATH = Path(__file__).parent / ".cache"
ruta_embedding_test = '../data_processed/test/test_w_embedding.json'

def getModel(config: RetrievalExpsConfig):
    model_id = config.model_name

    if config.quantization == True:
        model_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )
    else:
        model_config = None
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = AutoModelForCausalLM.from_pretrained(model_id, config=model_config)
    return model, tokenizer, device


def generate_text(model, tokenizer, prompt, device,
                  max_length=160000,
                  temperature=0.8,
                  num_return_sequences=1):
    t0 = time.time()
    # Tokenize the prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)  # Move input_ids to the same device as the model
    # Generate text
    output = model.generate(
        input_ids=input_ids,
        max_length=max_length,
        temperature=temperature,
        num_return_sequences=num_return_sequences,
        pad_token_id=tokenizer.eos_token_id,  # Set pad token to end of sequence token
        do_sample=True
    )
    # Decode the generated output
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

   # Split the generated text based on the prompt and take the portion after it
    generated_text = generated_text.split(prompt)[-1].strip()
    t_elapsed = time.time() - t0

    return generated_text, t_elapsed


def generate_embeddings_pipeline(logger: logging.Logger) -> str:
    """
    Pipeline para generar las respuestas a las preguntas médicas. Devuelve el tiempo que ha tardado en
    generar el índice (p.ej. para logearlo en mlflow)
    """
    logger.info("Cargando las preguntas de test..")
    test_answers = load_test_queries()

    logger.info("Generando los embeddings de las respuestas de test")
    t0 = time.time()
    for answer in test_answers:
        ans = answer['ANSWER']
        answer_embedding = get_sentence_embedding(ans)
        answer['EMBEDDING']= answer_embedding
    with open(ruta_embedding_test, 'w', encoding='utf-8') as file:
        json.dump(test_answers, file, ensure_ascii=False, indent=4)
    t_elapsed = time.time() - t0
    logger.info(f"Se han generado y guardado los embeddings en {t_elapsed:.0f} segundos.")


if __name__ == "__main__":

    # Configuramos el logging
    logger = colorlog.getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler = colorlog.StreamHandler()
    handler.setFormatter(
        colorlog.ColoredFormatter(
            "%(asctime)s-%(name)s-%(log_color)s%(levelname)s%(reset)s: %(message)s", datefmt="%y/%m/%d-%H:%M:%S"
        )
    )
    logger.addHandler(handler)

    inference_mode = False
    inference = input("Quiere realizar inferencia (i) o evaluación (e)? Responda i/e: ")
    if inference.lower() == "i":
        inference_mode=True
    else:
        inference_mode=False

    if inference_mode == False:
        # Configuramos mlflow
        mlflow.set_tracking_uri("http://localhost:8080")
        # Check that there is a running tracking server on the given URI
        try:
            response = requests.get("http://localhost:8080")
        except requests.exceptions.ConnectionError:
            logger.error("No se ha podido conectar con el servidor de mlflow. ¿Está arrancado?")
            sys.exit(1)

        mlflow.set_experiment("Embeddings Retrieval")

        with mlflow.start_run():

            # Cargamos la configuración del experimento y logeamos a mlflow
            exp_config = RetrievalExpsConfig()
            mlflow.log_params(exp_config.exp_params)

            # Cargamos datos de train y de test
            medical_data = get_medical_data()
            test_queries = load_test_queries()
            logger.info(f"Evaluando el modelo con {len(test_queries):,} queries...")

            if not os.path.isfile(ruta_embedding_test):
                generate_embeddings_pipeline(logger)

            # Cargamos el modelo
            model, tokenizer, device= getModel(exp_config)

            # Comenzamos el loop de evaluación
            n= 0
            mean_cosine, mean_rouge, accum_time = 0.0, 0.0, 0.0
            cosine = []
            rouge = []
            for question in tqdm(test_queries):
                query, expected_response = question["MESSAGE"], question["ANSWER"]
                prompt = generatePrompt(medical_data, query)
                response, t_elapsed = generate_text(model, tokenizer, prompt, device)
                accum_time += t_elapsed ##
                cosine_sim, rouge_score = getMetrics(expected_response, response)
                cosine.append(cosine_sim)
                rouge.append(rouge_score)
                n+=1
            mean_cosine = sum(cosine) / len(cosine)
            mean_rouge = sum(rouge) / len(rouge)

            logger.info(f"Cosine Similarity: {mean_cosine:.3f}, Rouge: {mean_rouge:.3f}")
            mlflow.log_metrics(
                {
                    "mean_cosine": round(mean_cosine, 3),
                    "mean_rouge": round(mean_rouge * 100, 1),
                    "secs_per_query": round(accum_time / n, 2),
                }
            )
            ranks_fig = plot_rank_distribution(ranks)
            mlflow.log_figure(ranks_fig, "rank_distribution.png")

    else:
        exp_config = RetrievalExpsConfig()
        id_sesion = str(datetime.now())
        user_queries = preprocesado()
        logger.info(f"Cargando el índice con los embeddings..")
        embedder = load_embedder(exp_config)
        embedder.show_progress = False
        index = FAISS.load_local(
            CACHE_PATH / f"faiss_{exp_config.index_config_unique_id}",
            embeddings=embedder,
            allow_dangerous_deserialization=True,
        )
        docs_for_user, t_elapsed = retrieval(user_queries, index, exp_config, logger)
        retrieved_data_for_user ={}
        for doc in docs_for_user:
            retrieved_data_for_user[doc.metadata["movie_id"]] = {}
            retrieved_data_for_user[doc.metadata["movie_id"]]["title_es"]=doc.metadata["title_es"]
            retrieved_data_for_user[doc.metadata["movie_id"]]["country"]=doc.metadata["country"]
            retrieved_data_for_user[doc.metadata["movie_id"]]["cast_top_5"]=doc.metadata["cast_top_5"]
        
        print(f"Le recomendamos estas películas:\n {retrieved_data_for_user}")
        satisfaccion = input("Indique su grado de satisfacción, por favor, nos ayuda a mejorar. Puntúe del 0 al 10: ")
        satisfaccion = int(satisfaccion)
        postprocesado(retrieved_data_for_user, satisfaccion, user_queries, id_sesion)