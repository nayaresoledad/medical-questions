import json
import random
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics.pairwise import cosine_similarity
from rouge import Rouge
from transformers import BertTokenizer, BertModel
import torch
import numpy as np


def load_test_queries(ruta: str = "./data_processed/test/test.json"):
    with open(ruta, "r") as f:
        queries = json.load(f)
    return queries

def get_sentence_embedding(sentence):
    # Cosine Similarity con BERT
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    inputs = tokenizer(sentence, padding=True, truncation=True, max_length=512, return_tensors='pt')
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze()

def getMetrics(test_answer, test_embedding, generated_answer, generated_answer_embedding):
    #Cosine Similarity
    cosine_sim = cosine_similarity([test_embedding.detach().numpy()], [generated_answer_embedding.detach().numpy()])[0][0]
    # ROUGE Score
    rouge = Rouge()
    rouge_score = rouge.get_scores(generated_answer, test_answer)
    return cosine_sim, rouge_score
