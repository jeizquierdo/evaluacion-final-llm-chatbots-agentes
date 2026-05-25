"""
agents/workflow.py

Builds and compiles the main LangGraph graph for the Academic Assistant.

General flow:
    guard → [classifier | finalizer]
              ↓
          classifier → researcher → [plannify | explain | summarize]  (fan-out)
                                          ↓
                                       validator
                                          ↓
                              [retry → task agents | finalizer]

Node responsibilities:
    guard       — Detects greetings and short-circuits the academic pipeline.
    classifier  — Identifies which academic tasks the user is requesting.
    researcher  — Gathers context via web search and Wikipedia.
    plannify    — Builds a personalised study plan.
    explain     — Produces a structured concept explanation.
    summarize   — Produces a structured topic summary.
    validator   — Evaluates output quality and triggers retries if needed.
    finalizer   — Assembles and returns the final response to the user.
"""

from functools import partial

from langgraph.graph import StateGraph, END

from agents.state      import AcademicState
from agents.guard      import guard_node, route_after_guard
from agents.classifier import classify_node_function
from agents.researcher import researcher_node_function, route_tasks_researcher
from agents.planner    import planner_node_function
from agents.explainer  import explainer_node_function
from agents.summarizer import summarizer_node_function
from agents.validator  import validator_node_function, route_tasks_validator
from agents.finalize   import finalizer_node_function


def build_graph(llm):
    """
    Builds and compiles the main Academic Assistant graph.

    Each node is a function that receives the shared AcademicState and returns
    a dict with the fields to update. The LLM is injected into every node that
    needs it via functools.partial, which is safer than lambdas in loops.

    Args:
        llm: Initialised LLM instance (ChatOllama, ChatOpenAI, etc.)

    Returns:
        CompiledGraph ready to be invoked from app.py.

    Usage:
        graph  = build_graph(config.llm)
        result = graph.invoke({"messages": [HumanMessage(content=user_input)]})
        answer = result["final_response"]
    """

    graph = StateGraph(AcademicState)

    # -----------------------------------------------------------------------
    # 1. NODES
    #
    #    Every node receives the full AcademicState and returns a partial dict
    #    that LangGraph merges back into the state using the reducers defined
    #    in state.py (add_messages for messages, dedup_add for completed_tasks).
    #
    #    partial() is used to pre-bind the llm argument so each node function
    #    still has the signature (state: dict) -> dict that LangGraph expects.
    # -----------------------------------------------------------------------

    # Guard — intercepts greetings before they reach the academic pipeline.
    # No LLM needed: pure regex matching.
    graph.add_node("guard", guard_node)

    # Classifier — reads the user message and returns detected_tasks.
    # Uses datetime tools (get_current_datetime, days_until) to resolve
    # temporal references before deciding which tasks to activate.
    graph.add_node("classifier", partial(classify_node_function, llm=llm))

    # Researcher — gathers external context for the detected tasks.
    # Uses web_search_tool and wikipedia_tool inside its own tool loop.
    graph.add_node("researcher", partial(researcher_node_function, llm=llm))

    # Task agents — run in parallel via LangGraph's Send API (fan-out).
    # Node names must exactly match the task strings returned by the classifier
    # ("plannify", "explain", "summarize") so that route_tasks_researcher
    # and route_tasks_validator can address them correctly via Send.
    graph.add_node("plannify",  partial(planner_node_function,    llm=llm))
    graph.add_node("explain",   partial(explainer_node_function,  llm=llm))
    graph.add_node("summarize", partial(summarizer_node_function, llm=llm))

    # Validator — evaluates each non-null output against quality criteria.
    # Uses analyze_output_quality and count_sections tools.
    # Increments retry_counts and sets validation_status ("ok"/"retry"/"forced_ok").
    graph.add_node("validator", partial(validator_node_function,  llm=llm))

    # Finalizer — merges all outputs into the final user-facing response.
    # Handles both the "greet" case and the academic tasks case.
    graph.add_node("finalizer", partial(finalizer_node_function,  llm=llm))

    # -----------------------------------------------------------------------
    # 2. ENTRY POINT
    #
    #    Every invocation starts at the guard node, which decides whether
    #    to short-circuit to the finalizer or continue to the classifier.
    # -----------------------------------------------------------------------

    graph.set_entry_point("guard")

    # -----------------------------------------------------------------------
    # 3. FIXED EDGES
    #
    #    Unconditional connections between nodes.
    # -----------------------------------------------------------------------

    # Classifier always hands off to the researcher once tasks are detected.
    graph.add_edge("classifier", "researcher")

    # Each task agent always feeds its output into the validator.
    # Because they run in parallel, all three edges must be declared
    # so LangGraph knows to wait for every active branch before validating.
    graph.add_edge("plannify",  "validator")
    graph.add_edge("explain",   "validator")
    graph.add_edge("summarize", "validator")

    # Finalizer always ends the graph — no further processing after this.
    graph.add_edge("finalizer", END)

    # -----------------------------------------------------------------------
    # 4. CONDITIONAL EDGES
    #
    #    Routing functions return either a string key or a list of Send objects.
    #    When using Send, no mapping dict is needed — Send addresses nodes
    #    directly by name. A mapping is only required when the routing function
    #    returns a plain string.
    # -----------------------------------------------------------------------

    # Guard → classifier (academic message) or finalizer (greeting).
    # route_after_guard returns a plain string, so a mapping is required.
    graph.add_conditional_edges(
        "guard",
        route_after_guard,
        {
            "classifier": "classifier",
            "finalizer":  "finalizer",
        }
    )

    # Researcher → fan-out to the detected task agents via Send.
    # route_tasks_researcher always returns a list of Send objects,
    # so no mapping dict is needed.
    graph.add_conditional_edges(
        "researcher",
        route_tasks_researcher,
        # {
        #     "summarize": "summarize",     # add this lines if you want to visualize the compiled graph with mermaid,
        #     "plannify": "plannify",       # because send objects are not rendered in the mermaid graph   
        #     "explain": "explain",
        # }
    )

    # Validator → retry loop or finalizer.
    # route_tasks_validator returns:
    #   - A list of Send objects when status is "retry" — no mapping needed.
    #   - The string "finalize" when status is "ok" or "forced_ok" — mapping required.
    # Only the string case needs to be listed in the mapping dict.
    graph.add_conditional_edges(
        "validator",
        route_tasks_validator,
        {
            "finalizer": "finalizer",
            # "summarize": "summarize",     # add this lines if you want to visualize the compiled graph with mermaid,
            # "plannify": "plannify",       # because send objects are not rendered in the mermaid graph 
            # "explain": "explain"
        }
    )

    # -----------------------------------------------------------------------
    # 5. COMPILE
    #
    #    graph.compile() validates the graph structure (no orphan nodes,
    #    no missing edge targets) and returns an executable CompiledGraph.
    # -----------------------------------------------------------------------

    return graph.compile()