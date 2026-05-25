"""
app.py

Entry point for the Academic Assistant Streamlit application.

Responsibilities:
  - Render the chat interface and sidebar configuration
  - Capture user input and invoke the LangGraph graph via stream()
  - Display live progress updates inline (replaced by final answer on completion)
  - Maintain conversation history across turns using st.session_state

Graph streaming strategy:
  graph.stream() with stream_mode="updates" yields one dict per node execution.
  Each dict has the shape { node_name: state_updates }.
  This lets us map node names to human-readable status messages and show them
  to the user in real time as the graph traverses each step.
"""

import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage

from utils.config import settings as config
from utils.utils import get_llm
from agents.workflow import build_graph


# ---------------------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Academic Assistant",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# CUSTOM CSS — clean dark-academic aesthetic
# ---------------------------------------------------------------------------

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700&family=DM+Sans:wght@300;400;500&display=swap');

  /* ── Global ── */
  html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
  }

  /* ── Background ── */
  .stApp {
    background: #0f1117;
    color: #e8e3d9;
  }

  /* ── Sidebar ── */
  [data-testid="stSidebar"] {
    background: #161b27 !important;
    border-right: 1px solid #2a2f3d;
  }
  [data-testid="stSidebar"] * {
    color: #c9c3b8 !important;
  }

  /* ── Header ── */
  .aa-header {
    text-align: center;
    padding: 2rem 0 1.5rem;
    border-bottom: 1px solid #2a2f3d;
    margin-bottom: 1.5rem;
  }
  .aa-header h1 {
    font-family: 'Playfair Display', serif;
    font-size: 2.1rem;
    font-weight: 700;
    color: #f5f0e8;
    letter-spacing: -0.5px;
    margin: 0;
  }
  .aa-header p {
    color: #7a8099;
    font-size: 0.88rem;
    margin: 0.4rem 0 0;
    font-weight: 300;
  }

  /* ── Chat bubbles ── */
  .msg-user {
    display: flex;
    justify-content: flex-end;
    margin: 0.8rem 0;
  }
  .msg-user .bubble {
    background: #1e4d8c;
    color: #e8f0fc;
    padding: 0.75rem 1.1rem;
    border-radius: 18px 18px 4px 18px;
    max-width: 78%;
    font-size: 0.93rem;
    line-height: 1.5;
    box-shadow: 0 2px 8px rgba(30,77,140,0.3);
  }

  .msg-assistant {
    display: flex;
    justify-content: flex-start;
    margin: 0.8rem 0;
  }
  .msg-assistant .bubble {
    background: #1a1f2e;
    color: #ddd8ce;
    padding: 0.75rem 1.1rem;
    border-radius: 18px 18px 18px 4px;
    max-width: 88%;
    font-size: 0.93rem;
    line-height: 1.6;
    border: 1px solid #2a2f3d;
  }

  /* ── Progress log ── */
  .progress-container {
    background: #111520;
    border: 1px solid #2a2f3d;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin: 0.8rem 0;
    font-family: 'DM Sans', monospace;
    font-size: 0.82rem;
  }
  .progress-title {
    color: #7a8099;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 0.6rem;
    font-weight: 500;
  }
  .progress-step {
    display: flex;
    align-items: flex-start;
    gap: 0.5rem;
    padding: 0.25rem 0;
    color: #9ba3bf;
    animation: fadeIn 0.3s ease;
  }
  .progress-step.active {
    color: #c9b96e;
  }
  .progress-step.done {
    color: #5a9e7a;
  }
  .progress-step.error {
    color: #c06060;
  }
  .progress-icon {
    flex-shrink: 0;
    width: 16px;
  }
  @keyframes fadeIn {
    from { opacity: 0; transform: translateY(4px); }
    to   { opacity: 1; transform: translateY(0); }
  }

  /* ── Spinner ── */
  .spinner {
    display: inline-block;
    animation: spin 1s linear infinite;
  }
  @keyframes spin {
    from { transform: rotate(0deg); }
    to   { transform: rotate(360deg); }
  }

  /* ── Input area ── */
  .stTextArea textarea {
    background: #161b27 !important;
    border: 1px solid #2a2f3d !important;
    border-radius: 12px !important;
    color: #e8e3d9 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.93rem !important;
    resize: none !important;
  }
  .stTextArea textarea:focus {
    border-color: #3d5a99 !important;
    box-shadow: 0 0 0 2px rgba(61,90,153,0.2) !important;
  }

  /* ── Buttons ── */
  .stButton > button {
    background: #1e4d8c !important;
    color: #e8f0fc !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    padding: 0.5rem 1.4rem !important;
    transition: background 0.2s !important;
  }
  .stButton > button:hover {
    background: #2a5fa8 !important;
  }

  /* secondary / clear button */
  .stButton.secondary > button {
    background: transparent !important;
    border: 1px solid #2a2f3d !important;
    color: #7a8099 !important;
  }

  /* ── Selectbox ── */
  [data-testid="stSelectbox"] > div > div {
    background: #1a1f2e !important;
    border: 1px solid #2a2f3d !important;
    color: #e8e3d9 !important;
    border-radius: 8px !important;
  }

  /* ── Markdown inside bubbles ── */
  .msg-assistant .bubble h1,
  .msg-assistant .bubble h2,
  .msg-assistant .bubble h3 {
    color: #f0e8d0;
    font-family: 'Playfair Display', serif;
    margin-top: 1rem;
  }
  .msg-assistant .bubble code {
    background: #0f1117;
    border-radius: 4px;
    padding: 0.1rem 0.35rem;
    font-size: 0.85em;
    color: #c9b96e;
  }
  .msg-assistant .bubble hr {
    border-color: #2a2f3d;
    margin: 0.8rem 0;
  }
  .msg-assistant .bubble ul, .msg-assistant .bubble ol {
    padding-left: 1.2rem;
  }
  .msg-assistant .bubble a {
    color: #7aade0;
  }

  /* ── Divider ── */
  hr { border-color: #2a2f3d !important; }

  /* ── Hide streamlit chrome ── */
  /* Keep header visible so the sidebar toggle button remains accessible */
  #MainMenu { visibility: hidden; }
  footer { visibility: hidden; }
  /* Make the header bar transparent/minimal instead of hiding it entirely */
  header[data-testid="stHeader"] {
    background: transparent !important;
    box-shadow: none !important;
    border-bottom: none !important;
  }
  .block-container { padding-top: 1rem !important; }
</style>
""", unsafe_allow_html=True)

# Force the sidebar to be expanded on initial load.
# Streamlit's initial_sidebar_state can be overridden by the browser's
# stored preference; this JS clicks the expand button if the sidebar is
# currently collapsed, but only runs once on the first render.
st.markdown("""
<script>
(function() {
  function expandSidebar() {
    // The sidebar toggle button has data-testid="collapsedControl"
    // when the sidebar is collapsed. Clicking it expands it.
    var btn = window.parent.document.querySelector('[data-testid="collapsedControl"]');
    if (btn) { btn.click(); }
  }
  // Run after a short delay to let Streamlit finish rendering the DOM
  setTimeout(expandSidebar, 300);
})();
</script>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# SESSION STATE INITIALISATION
# ---------------------------------------------------------------------------

def init_session_state():
    """Initialise all session state keys on first run."""
    defaults = {
        "messages":      [],   # list of {"role": "user"|"assistant", "content": str}
        "graph":         None, # compiled LangGraph graph
        "model_name":    config.gemini_model,
        "is_processing": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()


# ---------------------------------------------------------------------------
# GRAPH FACTORY — cached per model selection
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def get_graph(model_name: str):
    """
    Builds and caches the LangGraph compiled graph for the given model.

    Cached with @st.cache_resource so the graph (and its LLM client) is
    only constructed once per model per session — not on every rerender.

    Args:
        model_name: The model string (Ollama model name or Gemini model name).

    Returns:
        CompiledGraph ready for .stream() calls.
    """
    llm = get_llm(model_name)
    return build_graph(llm)


# ---------------------------------------------------------------------------
# NODE → HUMAN-READABLE STATUS MESSAGES
# ---------------------------------------------------------------------------

# Maps LangGraph node names to (icon, label) tuples shown in the progress log.
# The icon is an emoji; the label is shown as the current activity.
NODE_STATUS = {
    "guard":      ("🛡️",  "Checking message type..."),
    "classifier": ("🔍",  "Identifying academic tasks..."),
    "researcher": ("📚",  "Gathering information..."),
    "plannify":   ("🗓️",  "Building study plan..."),
    "explain":    ("🧠",  "Explaining concept..."),
    "summarize":  ("📋",  "Creating summary..."),
    "validator":  ("✅",  "Validating outputs..."),
    "finalizer":  ("✍️",  "Composing final answer..."),
}

# Task names that run in parallel after the researcher
PARALLEL_TASKS = {"plannify", "explain", "summarize"}


def build_progress_html(steps: list[dict]) -> str:
    """
    Renders the inline progress log as an HTML block.

    Each step dict has:
      - icon  (str): emoji
      - label (str): description text
      - state (str): "active" | "done" | "error"

    Args:
        steps: Ordered list of step dicts to render.

    Returns:
        HTML string for st.markdown(..., unsafe_allow_html=True).
    """
    rows = ""
    for step in steps:
        css_class = f"progress-step {step['state']}"
        rows += f"""
        <div class="{css_class}">
          <span class="progress-icon">{step['icon']}</span>
          <span>{step['label']}</span>
        </div>"""

    return f"""
    <div class="progress-container">
      <div class="progress-title">⚙ Processing</div>
      {rows}
    </div>"""


# ---------------------------------------------------------------------------
# GRAPH STREAMING + PROGRESS TRACKING
# ---------------------------------------------------------------------------

def stream_graph_response(user_input: str, placeholder) -> str:
    """
    Invokes the graph with stream_mode="updates" and renders live progress
    into the given Streamlit placeholder.

    After streaming completes the placeholder is cleared and the final
    response string is returned so the caller can render it as a chat bubble.

    Args:
        user_input:  The raw text typed by the user.
        placeholder: A st.empty() placeholder for the progress log.

    Returns:
        The final_response string from AcademicState, or an error message.
    """
    graph = get_graph(st.session_state["model_name"])

    # Build the full conversation history for the graph.
    # The graph uses add_messages reducer so we pass all prior turns plus
    # the new HumanMessage so the LLM has full context.
    history = []
    for msg in st.session_state["messages"]:
        if msg["role"] == "user":
            history.append(HumanMessage(content=msg["content"]))
        else:
            history.append(AIMessage(content=msg["content"]))

    # The current user message is already appended to st.session_state["messages"]
    # by the caller, so history already includes it. We construct the input
    # with just the messages; the graph manages all other state fields.
    graph_input = {"messages": history}

    # Progress steps accumulate here as nodes fire
    steps: list[dict] = []

    # Track which parallel tasks were launched so we can show them together
    parallel_launched: set[str] = set()
    parallel_step_index: int | None = None  # index in steps[] for the parallel entry

    final_response = ""

    try:
        # stream_mode="updates" yields {node_name: {state_field: value, ...}}
        # for every node that completes. We inspect the node name and state
        # updates to build meaningful progress messages.
        for chunk in graph.stream(graph_input, stream_mode="updates"):
            for node_name, updates in chunk.items():

                icon, base_label = NODE_STATUS.get(node_name, ("⚙️", node_name))

                # ── Guard node ───────────────────────────────────────────
                if node_name == "guard":
                    detected = updates.get("detected_tasks", [])
                    if detected == ["greet"]:
                        label = "Greeting detected — skipping academic pipeline"
                    else:
                        label = "Academic message confirmed"
                    steps.append({"icon": icon, "label": label, "state": "done"})

                # ── Classifier node ──────────────────────────────────────
                elif node_name == "classifier":
                    detected = updates.get("detected_tasks", [])
                    if detected:
                        task_labels = {
                            "plannify":  "study plan",
                            "explain":   "explanation",
                            "summarize": "summary",
                        }
                        names = [task_labels.get(t, t) for t in detected]
                        label = f"Tasks identified: {', '.join(names)}"
                    else:
                        label = base_label
                    steps.append({"icon": icon, "label": label, "state": "done"})

                # ── Researcher node ──────────────────────────────────────
                elif node_name == "researcher":
                    method = updates.get("search_method", "web_search")
                    method_labels = {
                        "web_search": "web search",
                        "rag":        "local documents",
                        "both":       "web + local documents",
                        "unknown":    "available sources",
                    }
                    label = f"Information gathered via {method_labels.get(method, method)}"
                    steps.append({"icon": icon, "label": label, "state": "done"})

                # ── Parallel task nodes (plannify / explain / summarize) ──
                elif node_name in PARALLEL_TASKS:
                    parallel_launched.add(node_name)

                    task_labels = {
                        "plannify":  "study plan",
                        "explain":   "explanation",
                        "summarize": "summary",
                    }

                    if parallel_step_index is None:
                        # First parallel task to arrive — create the entry
                        label = f"Running in parallel: {task_labels[node_name]}"
                        steps.append({"icon": "⚡", "label": label, "state": "active"})
                        parallel_step_index = len(steps) - 1
                    else:
                        # Update existing parallel entry with all tasks seen so far
                        names = [task_labels.get(t, t) for t in parallel_launched]
                        steps[parallel_step_index]["label"] = (
                            f"Completed in parallel: {', '.join(names)}"
                        )
                        steps[parallel_step_index]["state"] = "done"

                # ── Validator node ───────────────────────────────────────
                elif node_name == "validator":
                    # Mark parallel step as done if it wasn't already
                    if parallel_step_index is not None:
                        steps[parallel_step_index]["state"] = "done"

                    status = updates.get("validation_status", "ok")
                    failed = updates.get("failed_tasks", [])
                    counts = updates.get("retry_counts", {})

                    if status == "ok":
                        label = "All outputs validated successfully"
                        state = "done"
                    elif status == "forced_ok":
                        label = "Validation passed (retry limit reached)"
                        state = "done"
                    elif status == "retry" and failed:
                        task_labels = {
                            "plannify":  "study plan",
                            "explain":   "explanation",
                            "summarize": "summary",
                        }
                        failed_names = [task_labels.get(t, t) for t in failed]
                        retry_info = ", ".join(
                            f"{task_labels.get(t, t)} (attempt {counts.get(t, 1)})"
                            for t in failed
                        )
                        label = f"Retrying: {retry_info}"
                        state = "error"
                    else:
                        label = "Outputs validated"
                        state = "done"

                    steps.append({"icon": icon, "label": label, "state": state})

                # ── Finalizer node ───────────────────────────────────────
                elif node_name == "finalizer":
                    steps.append({"icon": icon, "label": "Composing response...", "state": "active"})
                    final_response = updates.get("final_response", "")

                # ── Any other node (future-proofing) ─────────────────────
                else:
                    steps.append({"icon": "⚙️", "label": node_name, "state": "done"})

                # Re-render the progress log after every node update
                placeholder.markdown(
                    build_progress_html(steps),
                    unsafe_allow_html=True,
                )

        # Mark the finalizer step as done once streaming is complete
        if steps and steps[-1]["icon"] == "✍️":
            steps[-1]["state"] = "done"
            steps[-1]["label"] = "Response ready"
            placeholder.markdown(
                build_progress_html(steps),
                unsafe_allow_html=True,
            )

    except Exception as e:
        steps.append({
            "icon":  "❌",
            "label": f"Error: {str(e)}",
            "state": "error",
        })
        placeholder.markdown(
            build_progress_html(steps),
            unsafe_allow_html=True,
        )
        final_response = f"Something went wrong: {str(e)}"

    return final_response


# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------

def render_sidebar():
    """Renders the model selector and conversation controls in the sidebar."""

    with st.sidebar:
        st.markdown("## 🎓 Academic Assistant")
        st.markdown("---")

        # ── Model selector ────────────────────────────────────────────────
        st.markdown("### Model")

        model_options = {
            f"✨ Gemini ({config.gemini_model})": config.gemini_model,
            f"🦙 Ollama ({config.default_model})": config.default_model,
        }

        # Disable Gemini if API key is not set
        gemini_available = config.is_gemini_available()
        if not gemini_available:
            model_options = {
                f"🦙 Ollama ({config.default_model})": config.default_model,
                "✨ Gemini (no API key)": config.gemini_model,
            }

        selected_label = st.selectbox(
            "Choose model",
            options=list(model_options.keys()),
            index=0,
            label_visibility="collapsed",
        )

        new_model = model_options[selected_label]

        if not gemini_available and new_model == config.gemini_model:
            st.warning("Set GOOGLE_API_KEY in .env to use Gemini.")
            new_model = config.default_model

        # If model changed, clear the cached graph so a new one is built
        if new_model != st.session_state["model_name"]:
            st.session_state["model_name"] = new_model
            st.rerun()

        st.markdown("---")

        # ── Capabilities info ─────────────────────────────────────────────
        st.markdown("### Capabilities")
        st.markdown("""
- 🧠 **Explain** concepts clearly
- 📋 **Summarize** topics & chapters
- 🗓️ **Plan** personalised study schedules
- 📚 **Search** web + your local docs
        """)

        st.markdown("---")

        # ── Clear conversation ────────────────────────────────────────────
        st.markdown("### Conversation")
        if st.button("🗑️ Clear chat", use_container_width=True):
            st.session_state["messages"] = []
            st.rerun()

        # ── Local docs hint ───────────────────────────────────────────────
        if st.session_state["messages"]:
            st.caption(f"{len(st.session_state['messages'])} messages in this session")

        st.markdown("---")
        st.caption("Add PDFs or TXTs to the `data/` folder to enable RAG search over your own documents.")


# ---------------------------------------------------------------------------
# CHAT HISTORY RENDERER
# ---------------------------------------------------------------------------

def render_chat_history():
    """Renders all messages stored in session state as styled chat bubbles."""

    for msg in st.session_state["messages"]:
        role = msg["role"]
        content = msg["content"]

        if role == "user":
            st.markdown(
                f'<div class="msg-user"><div class="bubble">{content}</div></div>',
                unsafe_allow_html=True,
            )
        else:
            # Assistant messages use st.markdown inside the bubble wrapper
            # so that markdown (bold, lists, headers, code) is rendered properly.
            # We split HTML wrapper and content to allow markdown rendering.
            with st.container():
                st.markdown('<div class="msg-assistant"><div class="bubble">', unsafe_allow_html=True)
                st.markdown(content)
                st.markdown('</div></div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# MAIN APP
# ---------------------------------------------------------------------------

def main():
    """Main application entry point."""

    render_sidebar()

    # ── Header ───────────────────────────────────────────────────────────
    st.markdown("""
    <div class="aa-header">
      <h1>Academic Assistant</h1>
      <p>Explain · Summarize · Plan — powered by multi-agent AI</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Welcome message (shown only when chat is empty) ───────────────────
    if not st.session_state["messages"]:
        st.markdown("""
        <div style="text-align:center; padding: 3rem 1rem; color: #5a6180;">
          <div style="font-size: 3rem; margin-bottom: 1rem;">🎓</div>
          <p style="font-size: 1rem; color: #7a8099; max-width: 480px; margin: 0 auto; line-height: 1.6;">
            Ask me to <strong style="color:#9ba3bf">explain</strong> a concept,
            <strong style="color:#9ba3bf">summarize</strong> a topic,
            or build a <strong style="color:#9ba3bf">study plan</strong> for you.
            <br><br>
            <em style="font-size: 0.85rem;">Try: "Explain gradient descent and give me a 3-week plan to learn it"</em>
          </p>
        </div>
        """, unsafe_allow_html=True)

    # ── Chat history ──────────────────────────────────────────────────────
    render_chat_history()

    # ── Input area ────────────────────────────────────────────────────────
    st.markdown("<div style='height: 1rem'></div>", unsafe_allow_html=True)

    col_input, col_send = st.columns([5, 1])

    with col_input:
        user_input = st.text_area(
            "Message",
            key="user_input",
            placeholder="Ask something academic...",
            height=80,
            label_visibility="collapsed",
        )

    with col_send:
        st.markdown("<div style='height: 28px'></div>", unsafe_allow_html=True)
        send_clicked = st.button("Send →", use_container_width=True, disabled=st.session_state["is_processing"])

    # ── Submit logic ──────────────────────────────────────────────────────
    # Trigger on button click or Ctrl+Enter (empty input guard)
    should_send = send_clicked and user_input and user_input.strip()

    if should_send and not st.session_state["is_processing"]:
        user_text = user_input.strip()

        # Append user message to history
        st.session_state["messages"].append({"role": "user", "content": user_text})
        st.session_state["is_processing"] = True
        st.rerun()

    # ── Processing turn ───────────────────────────────────────────────────
    # This block runs on the rerun triggered above, after the user message
    # is appended and is_processing is True.
    if st.session_state["is_processing"]:

        # Render a progress placeholder below the last user bubble
        progress_placeholder = st.empty()

        # Stream the graph and collect the final response
        final_response = stream_graph_response(
            user_input=st.session_state["messages"][-1]["content"],
            placeholder=progress_placeholder,
        )

        # Clear the progress log — replace with final answer
        progress_placeholder.empty()

        # Append assistant response to history
        if final_response:
            st.session_state["messages"].append({
                "role": "assistant",
                "content": final_response,
            })

        st.session_state["is_processing"] = False
        st.rerun()


if __name__ == "__main__":
    main()