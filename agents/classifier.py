"""agents.classifier

Classifier agent implementation: builds the classifier chain, executes any
requested tool calls (date helpers), and parses the model's JSON output into
the list of detected academic tasks.

This module exposes a LangGraph-compatible node function `classify_node_function`
and helper functions used internally to run tool loops and safely parse model
responses.
"""

from utils.config import settings as config
from utils.utils import load_prompt
from utils.utils import extract_content
from langchain_core.messages import AIMessage, ToolMessage
import json

# Import tools assigned to the classifier
from tools.datetime import get_current_datetime, days_until

# Tools the classifier is allowed to use
CLASSIFIER_TOOLS = [get_current_datetime, days_until]

# Map tool names to their callables for execution
TOOL_MAP = {tool.name: tool for tool in CLASSIFIER_TOOLS}


def create_classifier_agent(llm):
    """
    Creates the classifier chain.
    The LLM is bound with datetime tools so it can resolve temporal references
    in the user's message (e.g. "tengo el examen el viernes") before deciding
    which academic tasks to detect.
    """
    classifier_prompt = load_prompt(
        "prompts/classifier_prompt.txt",
        ["messages"],
        config.file_encoding
    )

    # bind_tools injects the tool schemas so the model can emit tool_call
    # blocks when it needs to resolve dates before classifying
    llm_with_tools = llm.bind_tools(CLASSIFIER_TOOLS)

    # Raw chain without StrOutputParser — we need the AIMessage intact
    # to inspect tool_calls before extracting the final JSON list
    classifier_chain = classifier_prompt | llm_with_tools

    return classifier_chain


def _run_tool_loop(chain, inputs: dict, max_iterations: int = 5) -> str:
    """
    Executes the classifier chain and resolves any tool calls the LLM emits.

    Flow:
      1. Invoke the chain → get an AIMessage
      2. If the message contains tool_calls, execute each one and feed
         the results back to the LLM as ToolMessages
      3. Repeat until the LLM returns a plain text response (the JSON list)
         or the iteration limit is reached

    Parameters
    ----------
    chain          : the compiled classifier chain (prompt | llm_with_tools)
    inputs         : the dict with prompt variables
    max_iterations : safety cap to avoid infinite loops
    """

    conversation_messages = []

    # First call — the LLM may answer directly or request tool calls
    response: AIMessage = chain.invoke(inputs)
    conversation_messages.append(response)

    for _ in range(max_iterations):

        # No tool calls → LLM produced its final JSON list, we are done
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
                result = f"Tool '{tool_name}' is not available for the classifier."

            tool_results.append(
                ToolMessage(content=str(result), tool_call_id=tool_id)
            )

        conversation_messages.extend(tool_results)

        # Feed tool results back to the LLM with full context so it can
        # finish classifying with the date information already resolved
        response = chain.invoke({
            **inputs,
            "messages": list(inputs.get("messages", [])) + conversation_messages,
        })
        conversation_messages.append(response)

    return extract_content(response)


def _parse_classifier_result(raw: str) -> list[str]:
    """
    Safely parses the JSON list output from the classifier LLM.

    The prompt instructs the model to return a JSON list of strings like:
      ["plannify", "explain"]

    If parsing fails, defaults to ["explain"] so the graph always has
    at least one task to route — consistent with the prompt's rule
    'nunca devuelvas una lista vacía'.
    """
    try:
        # Strip markdown code fences if the model included them
        clean = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        result = json.loads(clean)
        # Validate it's a non-empty list of strings
        if isinstance(result, list) and len(result) > 0:
            return result

        return ["explain"]

    except (json.JSONDecodeError, AttributeError):
        return ["explain"]


def classify_node_function(state: dict, llm) -> dict:
    """
    LangGraph node for the classifier agent.

    Builds the chain, runs the tool loop, parses the JSON list result,
    and returns detected_tasks to be merged into AcademicState.
    """
    classifier_chain = create_classifier_agent(llm)

    try:
        raw_result = _run_tool_loop(
            chain=classifier_chain,
            inputs={
                "messages": state.get("messages", []),
            },
        )
        detected_tasks = _parse_classifier_result(raw_result)
    except Exception as e:
        raise RuntimeError(f"classification failed: {str(e)}") from e

    return {"detected_tasks": detected_tasks}