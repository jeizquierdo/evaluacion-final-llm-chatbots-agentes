from utils.config import config
from utils.utils import load_prompt
from langchain_core.output_parsers import StrOutputParser


def create_finalizer_agent(llm):

    finalizer_prompt = load_prompt(
        "prompts/finalize_prompt.txt",
        ["messages", "detected_tasks", "explanation", "summary", "plan", "sources"],
        config.file_encoding
    )

    finalizer_chain = finalizer_prompt | llm | StrOutputParser()

    return finalizer_chain


def finalizer_node_function(state: dict, llm) -> dict:

    finalizer_chain = create_finalizer_agent(llm)

    try:
        final_response = finalizer_chain.invoke({
            "messages":       state.get("messages", []),
            "detected_tasks": state.get("detected_tasks", []),
            "explanation":    state.get("explanation", None),
            "summary":        state.get("summary", None),
            "plan":           state.get("plan", None),
            "sources":        state.get("sources", []),
        })
    except Exception as e:
        final_response = f"Finalization failed: {str(e)}"

    return {
        "final_response": final_response
    }