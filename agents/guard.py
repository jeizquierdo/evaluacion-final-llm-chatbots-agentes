import re

PATRON_SALUDO = re.compile(
    r"^(hola|hey|buenas?|hi|hello|good\s(morning|afternoon)|"
    r"cómo\sestás|qué\stal|gracias|thanks|ok|dale|perfecto)[!?.\s]*$",
    re.IGNORECASE
)

def guard_node(state: dict):
    texto = state["messages"][-1].content.strip()
    es_saludo = bool(PATRON_SALUDO.match(texto))
    return {"detected_tasks": ["saludo"] if es_saludo else []}

def route_after_guard(state: dict):
    if state["detected_tasks"] == ["saludo"]:
        return "finalizer"
    return "classifier" 