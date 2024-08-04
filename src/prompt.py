from __future__ import annotations

from src.data_util import Medical
import src.config


def generatePrompt(medical_data: list[Medical], user_query: str):
  B_INST, E_INST = "<s>[INST]", "[/INST]"
  B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
  DEFAULT_SYSTEM_PROMPT = """\
  You are an AI Medical Chatbot Assistant, provide comprehensive and informative responses to your inquiries.
  If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""
  context = "These are examples of how you must answer: "
  for item in medical_data:
      pregunta = item['MESSAGE']
      respuesta = item['ANSWER']
      context += f"Q: {pregunta}\nA: {respuesta}\n\n"
  SYSTEM_PROMPT = B_SYS + DEFAULT_SYSTEM_PROMPT + E_SYS + context
  instruction = f"User asks: {user_query}\n"
  prompt = B_INST + SYSTEM_PROMPT + instruction + E_INST

  return prompt