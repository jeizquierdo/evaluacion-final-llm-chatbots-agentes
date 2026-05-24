from utils.config import config
from utils.utils import load_prompt
from langchain_core.output_parsers import StrOutputParser

def create_summarizer_agent(llm):
    
    summarizer_prompt = load_prompt("prompts/summarizer_prompt.txt",["messages","detected_tasks","context","sources","validation_notes"], config.file_encoding)
    
    summarizer_chain = summarizer_prompt | llm | StrOutputParser()
    
    return summarizer_chain

def summarizer_node_function(state: dict, llm) -> dict:
     
    summarizer_chain = create_summarizer_agent(llm)
     
    try:
        summarizer_result = summarizer_chain.invoke({
            "messages": state.get("messages", []),
            "detected_tasks": state.get("detected_tasks", []),
            "context": state.get("context", ""),
            "sources": state.get("sources", []),
            "validation_notes": state.get("validation_notes", {}).get("summarize", "")
            })
    except Exception as e:
        summarizer_result = f"Summary failed: {str(e)}"
    
    return {
            "summary": summarizer_result,
            "completed_tasks": ["summary"] 
    } 