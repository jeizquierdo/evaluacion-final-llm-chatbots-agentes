import os
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun


try:
    from langchain_tavily import TavilySearch
    TAVILY_AVAILABLE = True
except ImportError:
    TAVILY_AVAILABLE = False


def build_web_search_tool():
    """
    """

    tavily_key = os.getenv("TAVILY_API_KEY", "")

    if TAVILY_AVAILABLE and tavily_key:
        search_engine = TavilySearch(
            max_results=5,
            topic="general",          
            include_answer=True,      
        )
        print("[web_search] Usando Tavily como motor de búsqueda.")
    else:
        search_engine = DuckDuckGoSearchRun()
        print("[web_search] Tavily no disponible. Usando DuckDuckGo como fallback.")

    return search_engine



@tool
def web_search_tool(query: str) -> str:
    """
    """
    engine = build_web_search_tool()
    try:
        result = engine.invoke(query)
        return result
    except Exception as e:
        return f"Web search error: {str(e)}"