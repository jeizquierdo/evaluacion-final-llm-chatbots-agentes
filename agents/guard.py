"""agents.guard

Guard node: lightweight pre-filter that detects greetings or social messages
that don't require the full academic pipeline. Provides routing helpers used
by the LangGraph graph to decide whether to short-circuit to the finalizer
or continue to the classifier.
"""

import re

# Pattern to detect greetings and social messages that don't require
# academic processing — these are routed directly to the finalizer
PATRON_SALUDO = re.compile(
    r"^(hola|hey|buenas?|hi|hello|good\s(morning|afternoon)|"
    r"cómo\sestás|qué\stal|gracias|thanks|ok|dale|perfecto)[!?.\s]*$",
    re.IGNORECASE
)


def guard_node(state: dict):
    """
    First node in the graph. Checks whether the user's message is a simple
    greeting or social phrase that doesn't need academic processing.

    Returns 'greet' (consistent with finalize_prompt.txt Case 1) when the
    message matches, or an empty list so the classifier takes over.
    """
    texto = state["messages"][-1].content.strip()
    es_saludo = bool(PATRON_SALUDO.match(texto))
    return {"detected_tasks": ["greet"] if es_saludo else []}


def route_after_guard(state: dict):
    """
    Conditional edge after the guard node.

    - If the message was a greeting → skip all agents and go straight
      to the finalizer, which handles the 'greet' case with a short reply.
    - Otherwise → send to the classifier for academic task detection.
    """
    if state["detected_tasks"] == ["greet"]:
        return "finalizer"
    return "classifier"