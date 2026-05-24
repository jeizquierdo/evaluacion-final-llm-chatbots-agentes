from utils.config import config
from utils.utils import load_prompt
from langchain_core.output_parsers import StrOutputParser

def create_planner_agent(llm):
    
    planner_prompt = load_prompt("prompts/planner_prompt.txt",["messages","detected_tasks","context","sources","validation_notes"], config.file_encoding)
    
    planner_chain = planner_prompt | llm | StrOutputParser()
    
    return planner_chain

def planner_node_function(state: dict, llm) -> dict:
     
    planner_chain = create_planner_agent(llm)
     
    try:
        planner_result = planner_chain.invoke({
            "messages": state.get("messages", []),
            "detected_tasks": state.get("detected_tasks", []),
            "context": state.get("context", ""),
            "sources": state.get("sources", []),
            "validation_notes": state.get("validation_notes", {}).get("plannify", "")
            })
    except Exception as e:
        planner_result = f"Planification failed: {str(e)}"
    
    return {
            "plan": planner_result,
            "completed_tasks": ["plan"] 
    } 