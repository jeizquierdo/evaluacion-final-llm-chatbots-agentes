from utils.config import config
from utils.utils import load_prompt
from langgraph.types import Send
from langchain_core.output_parsers import StrOutputParser

def create_validator_agent(llm):
    
    validator_prompt = load_prompt(
        "prompts/validator_prompt.txt",
        ["messages","detected_tasks","plan","explanation","summary","retry_counts"], 
        config.file_encoding)
    
    validator_chain = validator_prompt | llm | StrOutputParser()
    
    return validator_chain

def validator_node_function(state: dict, llm) -> dict:
     
    validator_chain = create_validator_agent(llm)
     
    try:
        validator_result = validator_chain.invoke({
            "messages": state.get("messages", []),
            "detected_tasks": state.get("detected_tasks", []),
            "plan": state.get("plan", ""),
            "explanation": state.get("explanation", ""),
            "summary": state.get("summary", ""),
            "retry_counts": state.get("retry_counts", {}),
            })
    except Exception as e:
        validator_result = f"Validation failed: {str(e)}"
    
    counts = dict(state.get("retry_counts", {}))
    for t in validator_result.get("failed_tasks", []):
        counts[t] = counts.get(t, 0) + 1
    
    return {
            "validation_status": validator_result["validation_status"],
            "validation_notes": validator_result["validation_notes"],
            "failed_tasks": validator_result["failed_tasks"],
            "retry_counts": counts
    }


def route_tasks_validator(state):
    if state.get("validation_status") == "retry":
        return [Send(f"{t}_agent", 
                    {
                        "message": state.get("messages",[]),
                        "context": state.get("context",""),
                        "sources": state.get("sources",[]),
                        "validation_notes": state.get("validation_notes", {}).get(t, ""),
                        "detected_tasks": state["detected_tasks"],
                    })  
                for t in state["failed_tasks"] if state.get("retry_counts", {}).get(t, 0) < config.max_retries] 
    else: return "finalize" 