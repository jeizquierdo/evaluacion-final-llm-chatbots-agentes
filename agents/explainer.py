from utils.config import config
from utils.utils import load_prompt
from langchain_core.output_parsers import StrOutputParser


def create_explainer_agent(llm):
    """
    Creates the explainer chain.
    Produces a structured explanation of the concept requested by the user,
    using the context gathered by the researcher.
    """
    explainer_prompt = load_prompt(
        "prompts/explainer_prompt.txt",
        ["messages", "detected_tasks", "context", "sources", "validation_notes"],
        config.file_encoding
    )

    explainer_chain = explainer_prompt | llm | StrOutputParser()

    return explainer_chain


def explainer_node_function(state: dict, llm) -> dict:
    """
    LangGraph node for the explainer agent.

    Returns explanation=None on failure so the validator and finalizer
    can distinguish between a missing output and a failed one, rather
    than receiving an error string that would be evaluated as content.
    """
    explainer_chain = create_explainer_agent(llm)

    try:
        explainer_result = explainer_chain.invoke({
            "messages":         state.get("messages", []),
            "detected_tasks":   state.get("detected_tasks", []),
            "context":          state.get("context", ""),
            "sources":          state.get("sources", []),
            "validation_notes": state.get("validation_notes", {}).get("explain", ""),
        })
    except Exception as e:
        raise RuntimeError(f"Explanation failed: {str(e)}") from e

    return {
        "explanation":     explainer_result,
        "completed_tasks": ["explain"],
    }