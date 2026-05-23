from utils.config import config
from utils.utils import load_prompt
from langchain_core.output_parsers import StrOutputParser


def create_classifier_agent(llm):
    
    classifier_prompt = load_prompt("prompts/classifier_prompt.txt",["messages"], config.file_encoding)
    
    classifier_chain =classifier_prompt | llm | StrOutputParser()
    
    return classifier_chain


def classify_node_function(state: dict, llm) -> dict:
     
    classify_chain = create_classifier_agent(llm)
     
    try:
        classify_result = classify_chain.invoke({"messages": state.get("messages", [])})
    except Exception as e:
        classify_result = f"Classification failed: {str(e)}"
    
    return {"detected_tasks": classify_result} 