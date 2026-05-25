"""tools.wikipedia

Wikipedia lookup utilities exposed as an LLM tool. The module prefers
Spanish articles but will fall back to English results when a concise
Spanish page isn't available. Returned text is raw article content.
"""

from langchain_core.tools import tool
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper


def build_wikipedia_tool(lang: str = "es", max_results: int = 2, doc_content_chars_max: int = 4000):
    """Create and return a configured Wikipedia query runner.

    Args:
        lang: language code ("es" or "en").
        max_results: number of pages to request.
        doc_content_chars_max: maximum characters to retrieve per doc.
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
    """Lookup a query on Wikipedia (prefers Spanish) and return article text.

    If Spanish results are short, the function will try English and prefer it
    when more informative.
    """
    tool_es = build_wikipedia_tool(lang="es")
    result = tool_es.invoke(query)


    if len(result.strip()) < 200:
        tool_en = build_wikipedia_tool(lang="en")
        result_en = tool_en.invoke(query)
        if len(result_en) > len(result):
            result = f"[Resultado en inglés — no se encontró versión en español]\n\n{result_en}"

    return result if result.strip() else f"No se encontraron resultados para: {query}"