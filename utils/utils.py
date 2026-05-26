"""utils.utils

Small utility helpers used by the application to load PromptTemplates and to
create language model clients (Ollama / Gemini). The functions are thin
wrappers around the underlying LangChain integrations and centralize encoding
and configuration handling.
"""

from utils.config import settings as config
from typing import Optional
from pathlib import Path
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI


def load_prompt(path: str, variables: list[str], encoding):
    
    text = Path(path).read_text(encoding=encoding)

    return PromptTemplate(
        input_variables=variables,
        template=text
    )
    
    
def _get_ollama_llm(temperature: Optional[float] = None):
    
    return ChatOllama(
        model=config.default_model,
        temperature=temperature or config.temperature,
        base_url=config.ollama_base_url
    )


def _get_gemini_llm(temperature: Optional[float] = None):

    if not config.is_gemini_available():
        raise ValueError(" Missing GOOGLE_API_KEY")
    
    return ChatGoogleGenerativeAI(
        model=config.gemini_model,
        temperature=temperature or config.temperature,
        google_api_key=config.google_api_key
    )
    
    
def get_llm(model: str ,temperature: Optional[float] = None):
    if model == config.gemini_model:
        return _get_gemini_llm(temperature)
    else: return _get_ollama_llm(temperature)    
    
    
def extract_content(response) -> str:
    """
    Extracts plain text from an LLM response.
    Handles both string content and list-of-blocks format
    returned by Anthropic and Gemini models.
    """
    content = response.content if hasattr(response, "content") else str(response)

    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                return block["text"]
        return str(content)

    return content    