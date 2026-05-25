import utils.config as config
from utils.utils import load_prompt
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.types import Send
import json

# Import tools assigned to the validator
from tools.text_analysis import analyze_output_quality, count_sections

# Tools the validator is allowed to use
VALIDATOR_TOOLS = [analyze_output_quality, count_sections]

# Map tool names to their callables for execution
TOOL_MAP = {tool.name: tool for tool in VALIDATOR_TOOLS}


def create_validator_agent(llm):
    """
    Creates the validator chain.
    The LLM is bound with text analysis tools so it can perform objective
    quality checks on each agent's output before deciding whether to approve
    or request a retry.
    """
    validator_prompt = load_prompt(
        "prompts/validator_prompt.txt",
        ["messages", "detected_tasks", "plan", "explanation", "summary", "retry_counts"],
        config.file_encoding
    )

    # bind_tools injects the tool schemas so the model can call
    # analyze_output_quality and count_sections before producing its verdict
    llm_with_tools = llm.bind_tools(VALIDATOR_TOOLS)

    # Raw chain without StrOutputParser — we need the AIMessage intact
    # to inspect tool_calls before extracting the final JSON verdict
    validator_chain = validator_prompt | llm_with_tools

    return validator_chain


def _run_tool_loop(chain, inputs: dict, max_iterations: int = 8) -> str:
    """
    Executes the validator chain and resolves any tool calls the LLM emits.

    The validator may call analyze_output_quality and count_sections once
    per output type (plan, explanation, summary), so max_iterations is set
    higher than other agents to allow up to 3 outputs x 2 tools each.

    Flow:
      1. Invoke the chain → get an AIMessage
      2. If the message contains tool_calls, execute each one and feed
         the results back to the LLM as ToolMessages
      3. Repeat until the LLM returns a plain text response (the JSON verdict)
         or the iteration limit is reached

    Parameters
    ----------
    chain          : the compiled validator chain (prompt | llm_with_tools)
    inputs         : the dict with prompt variables
    max_iterations : safety cap — set to 8 to allow multi-output validation
    """

    conversation_messages = []

    # First call — the LLM may answer directly or request tool calls
    response: AIMessage = chain.invoke(inputs)
    conversation_messages.append(response)

    for _ in range(max_iterations):

        # No tool calls → LLM produced its final JSON verdict, we are done
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
                result = f"Tool '{tool_name}' is not available for the validator."

            tool_results.append(
                ToolMessage(content=str(result), tool_call_id=tool_id)
            )

        conversation_messages.extend(tool_results)

        # Feed tool results back to the LLM so it can continue its evaluation
        # with the objective metrics already resolved
        response = chain.invoke({
            **inputs,
            "messages": list(inputs.get("messages", [])) + conversation_messages,
        })
        conversation_messages.append(response)

    return response.content if hasattr(response, "content") else str(response)


def _parse_validator_result(raw: str) -> dict:
    """
    Safely parses the JSON verdict output from the validator LLM.

    The prompt instructs the model to return a JSON object with three keys:
      - validation_status : "ok" | "retry" | "forced_ok"
      - validation_notes  : dict  e.g. {"plannify": "...", "explain": "..."}
      - failed_tasks      : list  e.g. ["plannify"]

    If parsing fails, returns a safe "forced_ok" fallback so the graph
    can always proceed to the finalizer rather than getting stuck.
    """
    try:
        # Strip markdown code fences if the model included them
        clean = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        result = json.loads(clean)

        # Validate required keys are present
        if "validation_status" not in result:
            raise ValueError("Missing validation_status key")

        # Ensure optional keys have safe defaults
        result.setdefault("validation_notes", {})
        result.setdefault("failed_tasks", [])

        return result

    except (json.JSONDecodeError, AttributeError, ValueError):
        # Fallback: approve everything so the graph never gets stuck
        return {
            "validation_status": "forced_ok",
            "validation_notes":  {},
            "failed_tasks":      [],
        }


def validator_node_function(state: dict, llm) -> dict:
    """
    LangGraph node for the validator agent.

    Builds the chain, runs the tool loop, parses the JSON verdict,
    updates retry_counts for any failed tasks, and returns the validation
    fields to be merged into AcademicState.
    """
    validator_chain = create_validator_agent(llm)

    try:
        raw_result = _run_tool_loop(
            chain=validator_chain,
            inputs={
                "messages":       state.get("messages", []),
                "detected_tasks": state.get("detected_tasks", []),
                "plan":           state.get("plan", ""),
                "explanation":    state.get("explanation", ""),
                "summary":        state.get("summary", ""),
                "retry_counts":   state.get("retry_counts", {}),
            },
        )
        validator_result = _parse_validator_result(raw_result)
    except Exception as e:
        raise RuntimeError(f"Validation error failed: {str(e)}") from e

    # Increment retry counter for each task that failed this round
    counts = dict(state.get("retry_counts", {}))
    for t in validator_result.get("failed_tasks", []):
        counts[t] = counts.get(t, 0) + 1

    return {
        "validation_status": validator_result["validation_status"],
        "validation_notes":  validator_result["validation_notes"],
        "failed_tasks":      validator_result["failed_tasks"],
        "retry_counts":      counts,
    }


def route_tasks_validator(state):
    """
    After validation, decides where the graph goes next.

    - "retry"               → fans out via Send to each failed task agent
                              (only if that task hasn't exceeded max_retries)
    - "ok" | "forced_ok"   → proceeds directly to the finalizer
    """
    if state.get("validation_status") == "retry":
        return [
            Send(
                f"{t}",
                {
                    "messages":          state.get("messages", []),
                    "context":          state.get("context", ""),
                    "sources":          state.get("sources", []),
                    "validation_notes": state.get("validation_notes", {}).get(t, ""),
                    "detected_tasks":   state["detected_tasks"],
                },
            )
            for t in state["failed_tasks"]
            if state.get("retry_counts", {}).get(t, 0) < config.max_retries
        ]
    else:
        return "finalizer"