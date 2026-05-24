"""
"""

from langchain_core.tools import tool
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper


def build_wikipedia_tool(lang: str = "es", max_results: int = 2, doc_content_chars_max: int = 4000):
    """
    """
    api_wrapper = WikipediaAPIWrapper(
        lang=lang,
        top_k_results=max_results,
        doc_content_chars_max=doc_content_chars_max,
    )
    wikipedia_tool = WikipediaQueryRun(api_wrapper=api_wrapper)
    return wikipedia_tool


@tool
def wikipedia_tool(query: str) -> str:
    """
    """
    tool_es = build_wikipedia_tool(lang="es")
    result = tool_es.invoke(query)


    if len(result.strip()) < 200:
        tool_en = build_wikipedia_tool(lang="en")
        result_en = tool_en.invoke(query)
        if len(result_en) > len(result):
            result = f"[Resultado en inglés — no se encontró versión en español]\n\n{result_en}"

    return result if result.strip() else f"No se encontraron resultados para: {query}"