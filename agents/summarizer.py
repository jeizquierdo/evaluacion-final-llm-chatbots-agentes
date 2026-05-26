"""agents.summarizer

Summarizer agent: produces structured summaries from the researcher's
context. Exposes `summarizer_node_function` and a helper to create the
LLM summarization chain.
"""

from utils.config import settings as config
from utils.utils import load_prompt
from langchain_core.output_parsers import StrOutputParser


def create_summarizer_agent(llm):
    """
    Creates the summarizer chain.
    Produces a structured summary of the topic requested by the user,
    using the context gathered by the researcher.
    """
    summarizer_prompt = load_prompt(
        "prompts/summarizer_prompt.txt",
        ["messages", "detected_tasks", "context", "sources", "validation_notes"],
        config.file_encoding
    )

    summarizer_chain = summarizer_prompt | llm | StrOutputParser()

    return summarizer_chain


def summarizer_node_function(state: dict, llm) -> dict:
    """
    LangGraph node for the summarizer agent.

    Returns summary=None on failure so the validator and finalizer
    can distinguish between a missing output and a failed one, rather
    than receiving an error string that would be evaluated as content.
    """
    summarizer_chain = create_summarizer_agent(llm)

    try:
        summarizer_result = summarizer_chain.invoke({
            "messages":         state.get("messages", []),
            "detected_tasks":   state.get("detected_tasks", []),
            "context":          state.get("context", ""),
            "sources":          state.get("sources", []),
            "validation_notes": state.get("validation_notes", {}).get("summarize", "") if isinstance(state.get("validation_notes"), dict) else ""
        })
    except Exception as e:
        raise RuntimeError(f"Summarization failed: {str(e)}") from e

    return {
        "summary": summarizer_result,
        "completed_tasks": ["summarize"],  
    }