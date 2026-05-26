"""
app.py

Entry point for the Academic Assistant Streamlit application.

Responsibilities:
  - Render the chat interface and sidebar configuration
  - Capture user input and invoke the LangGraph graph via invoke()
  - Display a simple "Thinking..." indicator while the graph runs
  - Maintain conversation history across turns using st.session_state
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

  /* ── Thinking indicator ── */
  .thinking {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    padding: 0.75rem 1.1rem;
    background: #1a1f2e;
    border: 1px solid #2a2f3d;
    border-radius: 18px 18px 18px 4px;
    color: #7a8099;
    font-size: 0.88rem;
    width: fit-content;
    margin: 0.8rem 0;
  }
  .thinking-dots span {
    display: inline-block;
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: #7a8099;
    animation: bounce 1.2s infinite;
  }
  .thinking-dots span:nth-child(2) { animation-delay: 0.2s; }
  .thinking-dots span:nth-child(3) { animation-delay: 0.4s; }
  @keyframes bounce {
    0%, 80%, 100% { transform: translateY(0); opacity: 0.4; }
    40%           { transform: translateY(-5px); opacity: 1; }
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
  /* Hide the Ctrl+Enter helper hint Streamlit injects below textareas */
  .stTextArea div[data-testid="InputInstructions"] {
    display: none !important;
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

  /* ── Selectbox ── */
  [data-testid="stSelectbox"] > div > div {
    background: #1a1f2e !important;
    border: 1px solid #2a2f3d !important;
    color: #e8e3d9 !important;
    border-radius: 8px !important;
  }

  /* ── Markdown inside assistant bubbles ── */
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

  /* ── Hide Streamlit chrome; keep header for sidebar toggle ── */
  #MainMenu { visibility: hidden; }
  footer    { visibility: hidden; }
  header[data-testid="stHeader"] {
    background:    transparent !important;
    box-shadow:    none !important;
    border-bottom: none !important;
  }
  .block-container { padding-top: 1rem !important; }
</style>
""", unsafe_allow_html=True)

# Force sidebar open on first load.
# Streamlit may restore the collapsed state from the browser; this JS
# clicks the expand toggle after the DOM settles if the bar is collapsed.
st.markdown("""
<script>
(function() {
  function expandSidebar() {
    var btn = window.parent.document.querySelector('[data-testid="collapsedControl"]');
    if (btn) { btn.click(); }
  }
  setTimeout(expandSidebar, 300);
})();
</script>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------------------------

def init_session_state():
    """Initialise all session-state keys on first run."""
    defaults = {
        "messages":      [],  # list of {"role": "user"|"assistant", "content": str}
        "model_name":    config.gemini_model,
        "is_processing": False,
        "input_key":     0,   # incremented after each send to reset the textarea
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
    Builds and caches the compiled LangGraph graph for the given model.
    Rebuilt only when the selected model changes.
    """
    llm = get_llm(model_name)
    return build_graph(llm)


# ---------------------------------------------------------------------------
# GRAPH INVOCATION
# ---------------------------------------------------------------------------

def run_graph(user_input: str) -> str:
    """
    Invokes the LangGraph graph synchronously and returns the final response.

    Builds the full conversation history (all prior turns + the new message)
    and passes it as the initial state. All other state fields are managed
    by the graph internally.

    Args:
        user_input: The latest message text (already appended to session state).

    Returns:
        The final_response string from AcademicState, or an error message.
    """
    graph = get_graph(st.session_state["model_name"])

    # Reconstruct the full message history as LangChain message objects
    history = []
    for msg in st.session_state["messages"]:
        if msg["role"] == "user":
            history.append(HumanMessage(content=msg["content"]))
        else:
            history.append(AIMessage(content=msg["content"]))

    try:
        result = graph.invoke({"messages": history})
        return result.get("final_response", "")
    except Exception as e:
        return f"Something went wrong: {str(e)}"


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

        gemini_available = config.is_gemini_available()

        if gemini_available:
            model_options = {
                f"✨ Gemini ({config.gemini_model})": config.gemini_model,
                f"🦙 Ollama ({config.default_model})": config.default_model,
            }
        else:
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

        # Rebuild the graph only when the model actually changes
        if new_model != st.session_state["model_name"]:
            st.session_state["model_name"] = new_model
            st.rerun()

        st.markdown("---")

        # ── Capabilities ──────────────────────────────────────────────────
        st.markdown("### Capabilities")
        st.markdown("""
- 🧠 **Explain** concepts clearly
- 📋 **Summarize** topics & chapters
- 🗓️ **Plan** personalised study schedules
- 📚 **Search** web + your local docs
        """)

        st.markdown("---")

        # ── Conversation controls ─────────────────────────────────────────
        st.markdown("### Conversation")
        if st.button("🗑️ Clear chat", use_container_width=True):
            st.session_state["messages"] = []
            st.rerun()

        if st.session_state["messages"]:
            st.caption(f"{len(st.session_state['messages'])} messages in this session")

        st.markdown("---")
        st.caption("Add PDFs or TXTs to the `data/` folder to enable RAG search over your own documents.")


# ---------------------------------------------------------------------------
# CHAT HISTORY
# ---------------------------------------------------------------------------

def render_chat_history():
    """Renders all messages stored in session state as styled chat bubbles."""

    for msg in st.session_state["messages"]:
        if msg["role"] == "user":
            st.markdown(
                f'<div class="msg-user"><div class="bubble">{msg["content"]}</div></div>',
                unsafe_allow_html=True,
            )
        else:
            # Split the HTML wrapper so Streamlit can render markdown inside it
            with st.container():
                st.markdown('<div class="msg-assistant"><div class="bubble">', unsafe_allow_html=True)
                st.markdown(msg["content"])
                st.markdown('</div></div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# MAIN APP
# ---------------------------------------------------------------------------

def main():
    """Main application entry point."""

    render_sidebar()

    # ── Page header ───────────────────────────────────────────────────────
    st.markdown("""
    <div class="aa-header">
      <h1>Academic Assistant</h1>
      <p>Explain · Summarize · Plan — powered by multi-agent AI</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Empty-state welcome ───────────────────────────────────────────────
    if not st.session_state["messages"]:
        st.markdown("""
        <div style="text-align:center; padding: 3rem 1rem;">
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

    # ── Thinking indicator (shown only while graph is running) ────────────
    if st.session_state["is_processing"]:
        st.markdown("""
        <div class="thinking">
          <span>Thinking</span>
          <span class="thinking-dots">
            <span></span><span></span><span></span>
          </span>
        </div>
        """, unsafe_allow_html=True)

    # ── Input area ────────────────────────────────────────────────────────
    st.markdown("<div style='height: 1rem'></div>", unsafe_allow_html=True)

    col_input, col_send = st.columns([5, 1])

    with col_input:
        user_input = st.text_area(
            "Message",
            key=f"user_input_{st.session_state['input_key']}",
            placeholder="Ask something academic...",
            height=80,
            label_visibility="collapsed",
        )

    with col_send:
        st.markdown("<div style='height: 28px'></div>", unsafe_allow_html=True)
        send_clicked = st.button(
            "Send →",
            use_container_width=True,
            disabled=st.session_state["is_processing"],
        )

    # ── Submit: capture input and trigger processing rerun ────────────────
    if send_clicked and user_input and user_input.strip() and not st.session_state["is_processing"]:
        st.session_state["messages"].append({"role": "user", "content": user_input.strip()})
        st.session_state["is_processing"] = True
        st.session_state["input_key"] += 1  # clears the textarea on next render
        st.rerun()

    # ── Processing: invoke the graph, then rerun to show the answer ───────
    if st.session_state["is_processing"]:
        final_response = run_graph(st.session_state["messages"][-1]["content"])

        if final_response:
            st.session_state["messages"].append({
                "role": "assistant",
                "content": final_response,
            })

        st.session_state["is_processing"] = False
        st.rerun()


if __name__ == "__main__":
    main()