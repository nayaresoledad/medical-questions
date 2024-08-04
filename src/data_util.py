import json
import psycopg
from src.schema import Medical

def get_medical_data() -> list[Medical]:
    """
    Returns all movies data from the movie_attributes table.
    Aplicamos unos filtros para reducir el número de películas y que sea más manejable para el ejercicio.
    """
    json_path = "./data_processed/train/train.json"
    with open(json_path, "r") as f:
        medical_data = json.load(f)

    medical_quest = [
        Medical(
            id=medical["ID"] if medical.get("ID") is not None else "",
            subject=medical["SUBJECT"] if medical.get("SUBJECT") is not None else "",
            message=medical["MESSAGE"] if medical.get("MESSAGE") is not None else "",
            answer=medical["ANSWER"] if isinstance(medical["ANSWER"], str) else medical["ANSWER"][0] if medical["ANSWER"] else "",
            focus=medical["FOCUS"] if isinstance(medical["FOCUS"], str) else medical["FOCUS"][0] if medical["FOCUS"] else "",
            type=medical["TYPE"] if isinstance(medical["TYPE"], str) else medical["TYPE"][0] if medical["TYPE"] else ""
        )
        for medical in medical_data
    ]
    return medical_quest