import json
import os

import psycopg
from dotenv import load_dotenv

from .schema import Medical

def get_medical_data() -> list[Medical]:
    """
    Returns all movies data from the movie_attributes table.
    Aplicamos unos filtros para reducir el número de películas y que sea más manejable para el ejercicio.
    """
    json_path = "/data_processed/train/train.json"
    with open(json_path, "r") as f:
        medical_data = json.load(f)

    medical_quest = [
        Medical(
            id=medical["ID"],
            subject=medical["SUBJECT"],
            message=medical["MESSAGE"],
            answer=medical["ANSWER"],
            focus=medical["FOCUS"],
            type=medical["TYPE"],
        )
        for medical in medical_data
    ]
    return medical_quest