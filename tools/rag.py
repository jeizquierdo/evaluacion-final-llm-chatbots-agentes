"""
tools/rag.py

LangChain tool that retrieves relevant context from local documents
stored in the data/ folder.

The tool checks whether documents are available before attempting retrieval,
returning a clear "no documents" message when the folder is empty so the
researcher agent knows to fall back to web search instead.
"""

from langchain_core.tools import tool
from tools.rag_loader import retrieve, data_folder_has_documents


@tool
def rag_tool(query: str) -> str:
    """
    Searches the local document knowledge base for information relevant to the query.

    Use this tool when the user refers to their own materials, notes, or uploaded
    documents (e.g. "mis apuntes", "el capítulo 3", "el PDF que subí").
    Do NOT use this tool for general knowledge questions — use web_search_tool
    or wikipedia_tool instead.

    Returns the most relevant text chunks found in the local documents,
    or a message indicating that no documents are available.

    Args:
        query: The search query to look up in the local knowledge base.

    Returns:
        str: Retrieved context from local documents, or a fallback message.
    """

    # Check upfront whether any documents exist in data/
    # This avoids building the index unnecessarily when the folder is empty
    if not data_folder_has_documents():
        return (
            "No hay documentos disponibles en la base de conocimiento local. "
            "No se puede realizar búsqueda RAG. "
            "Usá web_search_tool o wikipedia_tool para obtener información."
        )

    context, sources = retrieve(query)

    if not context:
        return (
            f"No se encontraron fragmentos relevantes para '{query}' "
            "en los documentos locales. "
            "Considerá usar web_search_tool para complementar."
        )

    # Format the result so the researcher can parse context and sources cleanly
    sources_text = "\n".join(f"- {s}" for s in sources)

    return (
        f"=== Resultados de documentos locales ===\n\n"
        f"{context}\n\n"
        f"=== Fuentes ===\n{sources_text}"
    )