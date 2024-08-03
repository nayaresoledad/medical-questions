import json

def cleanText(qna:str):
    '''Función que recibe un archivo json y limpia las preguntas y respuestas.
    -qna: ruta del archivo json'''
    with open(qna, 'r') as file:
       questions = json.load(file)

    for question in questions.values():
        
        quest = question['MESSAGE']
        if quest is not None:
            quest = quest.lower().strip()
            quest = re.sub(r'[\.\?\!\,\:\;\"]', '', quest)
            question['MESSAGE'] = quest

        answer = question['ANSWER']
        if isinstance(answer, list):
            anss = []
            for ans in answer:
                ans = ans.lower().strip()
                ans = re.sub(r'[\.\?\!\,\:\;\"]', '', ans)
                anss.append(ans)
            question['ANSWER'] = anss
        else:
            answer = answer.lower().strip()
            answer = re.sub(r'[\.\?\!\,\:\;\"]', '', answer)
            question['ANSWER'] = answer
        
            
    return questions

def clean_query_txt(query: str) -> str:
    query = query.replace("El usuario busca ", "").strip()
    query = query.replace("EE.UU", "Estados Unidos")
    query = query.lower()
    return query