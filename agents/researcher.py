from utils.config import config
from utils.utils import load_prompt
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.output_parsers import StrOutputParser
from langgraph.types import Send
import json

# Import tools assigned to the researcher
from tools.web_search import web_search_tool
from tools.wikipedia import wikipedia_tool

# Tools the researcher is allowed to use
RESEARCHER_TOOLS = [web_search_tool, wikipedia_tool]

# Map tool names to their callables for execution
TOOL_MAP = {tool.name: tool for tool in RESEARCHER_TOOLS}


def create_researcher_agent(llm):
    """
    Creates the researcher chain.
    The LLM is bound with the researcher tools so it can perform web searches
    and Wikipedia lookups before producing its structured JSON output.
    """
    researcher_prompt = load_prompt(
        "prompts/researcher_prompt.txt",
        ["messages", "detected_tasks"],
        config.file_encoding
    )

    # bind_tools injects the tool schemas into the request so the model
    # can emit tool_call blocks when it needs to search for information
    llm_with_tools = llm.bind_tools(RESEARCHER_TOOLS)

    # Raw chain without StrOutputParser — we need the AIMessage intact
    # to inspect tool_calls before extracting the final JSON text
    researcher_chain = researcher_prompt | llm_with_tools

    return researcher_chain


def _run_tool_loop(chain, inputs: dict, max_iterations: int = 5) -> str:
    """
    Executes the researcher chain and resolves any tool calls the LLM emits.

    Flow:
      1. Invoke the chain → get an AIMessage
      2. If the message contains tool_calls, execute each one and feed
         the results back to the LLM as ToolMessages
      3. Repeat until the LLM returns a plain text response (the JSON)
         or the iteration limit is reached

    Parameters
    ----------
    chain          : the compiled researcher chain (prompt | llm_with_tools)
    inputs         : the dict with prompt variables
    max_iterations : safety cap to avoid infinite loops
    """

    conversation_messages = []

    # First call — the LLM may answer directly or request tool calls
    response: AIMessage = chain.invoke(inputs)
    conversation_messages.append(response)

    for _ in range(max_iterations):

        # No tool calls → LLM produced its final JSON answer, we are done
        if not getattr(response, "tool_calls", None):
            break

        # Execute every tool the LLM requested in this turn
        tool_results: list[ToolMessage] = []
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args  = tool_call["args"]
            tool_id    = tool_call["id"]

            if tool_name in TOOL_MAP:
                try:
                    result = TOOL_MAP[tool_name].invoke(tool_args)
                except Exception as e:
                    result = f"Tool '{tool_name}' failed: {str(e)}"
            else:
                result = f"Tool '{tool_name}' is not available for the researcher."

            tool_results.append(
                ToolMessage(content=str(result), tool_call_id=tool_id)
            )

        conversation_messages.extend(tool_results)

        # Feed tool results back to the LLM with full context so it can
        # continue reasoning and eventually produce the final JSON
        response = chain.invoke({
            **inputs,
            "messages": list(inputs.get("messages", [])) + conversation_messages,
        })
        conversation_messages.append(response)

    return response.content if hasattr(response, "content") else str(response)


def _parse_researcher_result(raw: str) -> dict:
    """
    Safely parses the JSON output from the researcher LLM.

    The prompt instructs the model to return a JSON with three keys:
      - search_method : str
      - context       : str
      - sources       : list[str]

    If parsing fails for any reason, returns a safe fallback so the
    rest of the graph can continue without crashing.
    """
    try:
        # Strip markdown code fences if the model included them
        clean = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        return json.loads(clean)
    except (json.JSONDecodeError, AttributeError):
        # Fallback: treat the raw text as context so no information is lost
        return {
            "search_method": "unknown",
            "context": raw,
            "sources": [],
        }


def researcher_node_function(state: dict, llm) -> dict:
    """
    LangGraph node for the researcher agent.

    Builds the chain, runs the tool loop, parses the JSON result,
    and returns the fields that should be merged into AcademicState.
    """
    researcher_chain = create_researcher_agent(llm)

    try:
        raw_result = _run_tool_loop(
            chain=researcher_chain,
            inputs={
                "messages":       state.get("messages", []),
                "detected_tasks": state.get("detected_tasks", []),
            },
        )
        researcher_result = _parse_researcher_result(raw_result)
    except Exception as e:
        raise RuntimeError(f"Research failed: {str(e)}") from e

    return {
        "search_method": researcher_result["search_method"],
        "context":       researcher_result["context"],
        "sources":       researcher_result["sources"],
    }


def route_tasks_researcher(state):
    """
    After the researcher finishes, fans out to the task agents in parallel
    using LangGraph's Send API — one Send per detected task.
    """
    return [
        Send(
            f"{t}",
            {
                "messages":         state.get("messages", []),
                "context":          state.get("context", ""),
                "sources":          state.get("sources", []),
                "validation_notes": state.get("validation_notes", {}).get(t, ""),
                "detected_tasks":   state.get("detected_tasks", []),
            },
        )
        for t in state["detected_tasks"]
    ]