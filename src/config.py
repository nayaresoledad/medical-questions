from __future__ import annotations

from typing import Callable

from src.data_util import Medical
from src.retrieval_pipeline_utils import clean_query_txt


class RetrievalExpsConfig:
    """
    Class to keep track of all the parameters used in the embeddings experiments.
    Any attribute created in this class will be logged to mlflow.

    Nota: cuando definimos atributos de tipo Callable, debemos usar `staticmethod` para que la función pueda ser llamada
    s
    """

    def __init__(self):

        # Parámetros para la generación de embeddings

        self.model_name: str = "johnsnowlabs/JSL-MedLlama-3-8B-v2.0" # mistralai/Mistral-7B-Instruct-v0.2 #all-MiniLM-L6-v2 , mistralai/Mistral-7B-v0.1, TheBloke/Mistral-7B-Instruct-v0.1-AWQ, dariolopez/roberta-base-bne-finetuned-msmarco-qa-es-mnrl-mn, sentence-transformers/paraphrase-multilingual-mpnet-base-v2
        self.normalize_embeddings: bool = False  # Normalizar los embeddings a longitud 1 antes de indexarlos

        self._query_prepro_fn: Callable = clean_query_txt

        self.quantization: bool = False

    ## NO MODIFICAR A PARTIR DE AQUÍ ##

    def text_to_embed_fn(self, medical: Medical) -> str:
        return self._text_to_embed_fn(medical)

    def query_prepro_fn(self, query: dict) -> str:
        return self._query_prepro_fn(query)

    @property
    def index_config_unique_id(self) -> str:
        mname = self.model_name.replace("/", "_")
        return f"{mname}_{self._text_to_embed_fn.__name__}_{self.normalize_embeddings}"

    @property
    def exp_params(self) -> dict:
        """
        Return the config parameters as a dictionary. To be used, for example, in mlflow logging
        """
        return {
            "model_name": self.model_name,
            "normalize_embeddings": self.normalize_embeddings,
            "query_prepro_fn": self._query_prepro_fn.__name__,
        }