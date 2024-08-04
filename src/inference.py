import json
import time
from datetime import datetime
import logging
import os

from langchain_core.documents import Document

from data_util import Medical
from config import RetrievalExpsConfig
from langchain_community.vectorstores import FAISS



def preprocesado():
    querie_input = input("Por favor, describa su problema médico: ")
    query=str(querie_input)
    return query

def postprocesado(generated_answer: str, satisfaccion: int, query: str, id_sesion: str):
    sesiones = {}
    sesiones["query"]= query
    sesiones["generated_answer"]=generated_answer
    sesiones["satisfaccion"]= satisfaccion
    file_path = "../data/sesiones.json"
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            try:
                data = json.load(file)
            except json.JSONDecodeError:
                data = {}
    else:
        data={}

    data[id_sesion] = sesiones

    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)