from utils.config import config
from utils.utils import load_prompt
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.output_parsers import StrOutputParser

# Import tools assigned to the planner
from tools.wikipedia import wikipedia_tool
from tools.datetime import get_current_datetime, calculate_study_dates, days_until

# Tools the planner is allowed to use
PLANNER_TOOLS = [wikipedia_tool, get_current_datetime, calculate_study_dates, days_until]

# Map tool names to their callables for execution
TOOL_MAP = {tool.name: tool for tool in PLANNER_TOOLS}


def create_planner_agent(llm):
    """
    Creates the planner chain.
    The LLM is bound with the planner tools so it can request tool calls
    when it needs external information (dates, Wikipedia context, etc.).
    """
    planner_prompt = load_prompt(
        "prompts/planner_prompt.txt",
        ["messages", "detected_tasks", "context", "sources", "validation_notes"],
        config.file_encoding
    )

    # bind_tools tells the LLM which tools are available and injects
    # their schemas into the request so the model can emit tool_call blocks
    llm_with_tools = llm.bind_tools(PLANNER_TOOLS)

    # We do NOT attach StrOutputParser here because the raw AIMessage is
    # needed to inspect tool_calls before the final text is extracted
    planner_chain = planner_prompt | llm_with_tools

    return planner_chain


def _run_tool_loop(chain, inputs: dict, max_iterations: int = 5) -> str:
    """
    Executes the planner chain and resolves any tool calls the LLM emits.

    Flow:
      1. Invoke the chain → get an AIMessage
      2. If the message contains tool_calls, execute each one and feed
         the results back to the LLM as ToolMessages
      3. Repeat until the LLM returns a plain text response or the
         iteration limit is reached

    Parameters
    ----------
    chain        : the compiled planner chain (prompt | llm_with_tools)
    inputs       : the dict with prompt variables
    max_iterations: safety cap to avoid infinite loops
    """

    # conversation_messages accumulates the back-and-forth with the LLM
    # so it has full context when tool results are returned
    conversation_messages = []

    # First call — the LLM may answer directly or request tool calls
    response: AIMessage = chain.invoke(inputs)
    conversation_messages.append(response)

    for _ in range(max_iterations):

        # No tool calls → LLM produced its final answer, we are done
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
                result = f"Tool '{tool_name}' is not available for the planner."

            tool_results.append(
                ToolMessage(content=str(result), tool_call_id=tool_id)
            )

        conversation_messages.extend(tool_results)

        # Feed tool results back to the LLM so it can continue reasoning
        # We append to the original inputs messages list so the chain has
        # the full context (prompt + prior messages + tool results)
        response = chain.invoke({
            **inputs,
            # Extend the messages with the tool exchange so the LLM sees
            # both its own tool_call and the result that came back
            "messages": list(inputs.get("messages", [])) + conversation_messages,
        })
        conversation_messages.append(response)

    # Extract plain text from the final AIMessage
    return response.content if hasattr(response, "content") else str(response)


def planner_node_function(state: dict, llm) -> dict:
    """
    LangGraph node for the planner agent.

    Builds the chain, runs the tool loop, and returns the fields
    that should be merged into the shared AcademicState.
    """
    planner_chain = create_planner_agent(llm)

    try:
        planner_result = _run_tool_loop(
            chain=planner_chain,
            inputs={
                "messages":         state.get("messages", []),
                "detected_tasks":   state.get("detected_tasks", []),
                "context":          state.get("context", ""),
                "sources":          state.get("sources", []),
                "validation_notes": state.get("validation_notes", {}).get("plannify", ""),
            },
        )
    except Exception as e:
        raise RuntimeError(f"Plannification failed: {str(e)}") from e

    return {
        "plan": planner_result,
        "completed_tasks": ["plan"],
    }