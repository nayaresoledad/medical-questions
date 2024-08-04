from __future__ import annotations

from langchain_core.documents import Document

from data_util import Medical
import config


def create_docs_to_embedd(medical_quest: list[Medical], config: config.RetrievalExpsConfig) -> list[Document]:
    """
    Convierte una lista de objetos `Medical` a una lista the objetos `Document`(usada por Langchain).
    En esta función se decide que parte de los datos será usado como embeddings y que parte como metadata.
    """
    movies_as_docs = []
    for medical in medical_quest:
        content = config.text_to_embed_fn(medical)
        metadata = medical.model_dump()
        doc = Document(page_content=content, metadata=metadata)
        movies_as_docs.append(doc)

    return movies_as_docs

def get_subject_message_answer_type(medical_data: Medical) -> str:
    return f'Subject: {medical_data.subject}\n Message: {medical_data.message} Answer: {medical_data.answer}, Type: {medical_data.type}'