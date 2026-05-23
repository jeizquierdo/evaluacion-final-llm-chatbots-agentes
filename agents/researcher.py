from utils.config import config
from utils.utils import load_prompt
from langgraph.types import Send
from langchain_core.output_parsers import StrOutputParser

def create_researcher_agent(llm):
    
    researcher_prompt = load_prompt("prompts/researcher_prompt.txt",["messages"], config.file_encoding)
    
    researcher_chain =researcher_prompt | llm | StrOutputParser()
    
    return researcher_chain


def researcher_node_function(state: dict, llm) -> dict:
     
    researcher_chain = create_researcher_agent(llm)
     
    try:
        researcher_result = researcher_chain.invoke({"messages": state.get("messages", [])})
    except Exception as e:
        researcher_result = f"Classification failed: {str(e)}"
    
    return {
            "search_method": researcher_result["search_method"],
            "context": researcher_result["context"],
            "sources": researcher_result["sources"]
    } 
    
    
def route_tasks(state):
    return [Send(f"{t}_agent", {"task": t, "context": state["context"]}) 
            for t in state["detected_tasks"]]