from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, add_messages



def dedup_add(a: list, b: list) -> list:
    return list(dict.fromkeys(a + b))  # preserve order and remove duplicates


class AcademicState(TypedDict):

    # --- INPUT ---
    messages: Annotated[list[BaseMessage], add_messages]
    detected_tasks: list[str]

    # --- RESEARCH ---
    context: str
    sources: list[str]
    search_method: str

    # --- TASK RESULTS ---
    plan: str | None
    explanation: str | None
    summary: str | None
    completed_tasks: Annotated[list[str], dedup_add]

    # --- RETRY CONTROL ---
    retry_counts: dict[str, int]   # {"plannify": 1, "explain": 0, "summarize": 2}
    failed_tasks: list[str]        # Wich tasks have failed in this iteration

    # --- VALIDATION ---
    validation_status: str         # "ok" | "retry" | "forced_ok"
    validation_notes: dict         # {"plannify": "lack of structure", "summarize": "ok"}

    # --- OUTPUT ---
    final_response: str
    
    