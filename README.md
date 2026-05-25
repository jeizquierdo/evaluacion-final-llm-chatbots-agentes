# Academic Assistant — LLM Multi-Agent System

> Final project for the postgraduate course: *LLMs, Chatbots and Agents*

---

## Chosen use case: Academic Assistant

This project implements a multi-agent **Academic Assistant** powered by **Ollama / Gemini**, **LangGraph**, and **Streamlit**. Given a user query, the system can:

- **Explain** any academic concept in a structured, didactic format
- **Summarize** a topic, chapter, or subject area concisely
- **Plan** a personalised, date-aware study schedule

The assistant can search the web, consult Wikipedia, and retrieve information from local documents (PDFs, TXTs, DOCX) uploaded by the user.

---

## Architecture

The system is built around a **directed graph** (LangGraph `StateGraph`) where each node is a specialised agent with a single, clearly defined responsibility. Agents communicate exclusively through a shared typed state (`AcademicState`) — no agent calls another directly.

### Graph flow

```
User message
      │
      ▼
   [guard]
      │
      ├── greeting ──────────────────────────────► [finalizer] ──► Response
      │
      └── academic message
                │
                ▼
         [classifier]
                │
                ▼
          [researcher]
                │
          ┌─────┼─────┐          (parallel fan-out via Send API)
          ▼     ▼     ▼
      [plannify][explain][summarize]
          │     │     │
          └─────┼─────┘
                ▼
          [validator]
                │
                ├── retry ──► [plannify / explain / summarize]  (loop, max 3x)
                │
                └── ok / forced_ok
                          │
                          ▼
                     [finalizer] ──► Response
```

### Agents and their roles

| Agent | File | Responsibility |
|---|---|---|
| **Guard** | `agents/guard.py` | Detects greetings via regex and short-circuits the academic pipeline |
| **Classifier** | `agents/classifier.py` | Reads the user message and returns the list of detected tasks (`plannify`, `explain`, `summarize`) |
| **Researcher** | `agents/researcher.py` | Gathers context via web search, Wikipedia, and/or RAG over local documents |
| **Planner** | `agents/planner.py` | Builds a personalised, date-anchored study plan |
| **Explainer** | `agents/explainer.py` | Produces a structured concept explanation (definition, mechanism, example, related concepts) |
| **Summarizer** | `agents/summarizer.py` | Produces a concise structured summary (central idea, key points, conclusion) |
| **Validator** | `agents/validator.py` | Evaluates output quality using text analysis tools; triggers retries for failed outputs |
| **Finalizer** | `agents/finalize.py` | Merges all outputs into a single coherent user-facing response |

---

## Design decisions

### 1. Single shared state, no direct agent coupling
All agents read from and write to `AcademicState` (defined in `agents/state.py`). Agents are completely decoupled — the planner has no knowledge of the explainer and vice versa. This makes it easy to add, remove, or swap agents without touching others.

### 2. Parallel task execution via LangGraph's Send API
After the researcher finishes, the graph fans out to all detected task agents simultaneously using `Send` objects. A query like *"explain gradient descent and give me a plan"* runs the explainer and planner in parallel, reducing total latency.

### 3. Validation loop with retry cap
The validator agent uses two objective tools (`analyze_output_quality`, `count_sections`) to evaluate each output against structural and length criteria. If an output fails, it is re-queued to the corresponding agent. A per-task retry counter prevents infinite loops — after `MAX_RETRIES` attempts, the validator issues `forced_ok` and the graph proceeds anyway.

### 4. Prompts separated from code
All agent prompts live in the `prompts/` folder as plain `.txt` files with `{variable}` placeholders. This makes it easy to iterate on prompt design without touching Python code.

### 5. Tool injection via `bind_tools`
Agents that need external tools (web search, Wikipedia, datetime helpers, text analysis) receive them through LangChain's `bind_tools` mechanism. Each agent only has access to the tools it actually needs, keeping tool namespaces small and reducing the risk of the LLM calling the wrong tool.

### 6. Dual LLM support (Ollama + Gemini)
The application supports two backends selectable from the sidebar. Gemini (`gemini-2.5-flash-lite`) is used when `GOOGLE_API_KEY` is set; otherwise the system falls back to a local Ollama model (`llama3.2:3b`). The graph is rebuilt once per model selection and cached via `@st.cache_resource`.

### 7. RAG over local documents
Users can drop PDF, TXT, or DOCX files into the `data/` folder. On first retrieval the system builds a FAISS index using `sentence-transformers/all-MiniLM-L6-v2` embeddings (no API key needed) and caches it for the session. If no documents are present, the researcher agent falls back to web search automatically.

---

## Repository structure

```
.
├── app.py                  # Streamlit entry point and chat interface
├── .env.example            # Environment variable template
│
├── agents/
│   ├── state.py            # Shared AcademicState TypedDict + reducers
│   ├── workflow.py         # LangGraph graph construction and compilation
│   ├── guard.py            # Greeting detection and routing
│   ├── classifier.py       # Task classification agent
│   ├── researcher.py       # Context gathering agent (web + RAG)
│   ├── planner.py          # Study plan generation agent
│   ├── explainer.py        # Concept explanation agent
│   ├── summarizer.py       # Topic summarization agent
│   ├── validator.py        # Output quality validation agent
│   └── finalize.py         # Response assembly agent
│
├── prompts/
│   ├── classifier_prompt.txt
│   ├── researcher_prompt.txt
│   ├── planner_prompt.txt
│   ├── explainer_prompt.txt
│   ├── summarizer_prompt.txt
│   ├── validator_prompt.txt
│   └── finalize_prompt.txt
│
├── tools/
│   ├── web_search.py       # Tavily (primary) / DuckDuckGo (fallback) search
│   ├── wikipedia.py        # Wikipedia lookup (ES with EN fallback)
│   ├── rag.py              # Local document retrieval tool
│   ├── rag_loader.py       # FAISS index construction and caching
│   ├── datetime.py         # Date/time helpers for planning agents
│   └── text_analysis.py    # Output quality analysis tools for the validator
│
├── utils/
│   ├── config.py           # Settings dataclass loaded from .env
│   └── utils.py            # LLM factory and prompt loader helpers
│
├── data/                   # Drop PDF / TXT / DOCX files here to enable RAG
│
└── README.md
```

---

## Setup and installation

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com/) installed and running locally (for local model usage)
- A Google API key (optional, for Gemini)
- A Tavily API key (optional, for better web search)

### 1. Clone and install dependencies

```bash
git clone <your-fork-url>
cd evaluacion-final-llm-chatbots-agentes
pip install -r requirements.txt
```

### 2. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` with your values:

```env
GOOGLE_API_KEY=your-google-api-key-here     # optional — enables Gemini
TAVILY_API_KEY=your-tavily-api-key-here     # optional — better web search
OLLAMA_BASE_URL=http://localhost:11434
DEFAULT_MODEL=llama3.2:3b
GEMINI_MODEL=gemini-2.5-flash-lite
MAX_RETRIES=3
FILE_ENCODING=utf-8
TEMPERATURE=0.7
MAX_TOKENS=2048
```

### 3. Pull the Ollama model (if using local LLM)

```bash
ollama pull llama3.2:3b
```

### 4. (Optional) Add documents for RAG

Drop any `.pdf`, `.txt`, or `.docx` files into the `data/` folder. The system will index them automatically on first query.

### 5. Run the application

```bash
streamlit run app.py
```

---

## Tools available per agent

| Tool | Used by |
|---|---|
| `web_search_tool` | Researcher |
| `wikipedia_tool` | Researcher, Planner |
| `rag_tool` | Researcher |
| `get_current_datetime` | Classifier, Planner |
| `days_until` | Classifier, Planner |
| `calculate_study_dates` | Planner |
| `analyze_output_quality` | Validator |
| `count_sections` | Validator |

---

## Example queries

```
Explain gradient descent and give me a 3-week study plan for it
```
```
Summarize the topic of neural networks
```
```
I have an exam on June 15th, help me plan how to study linear algebra from scratch
```
```
Explain what a transformer is and summarize the main concepts
```

---

## Minimum requirements checklist

- [x] At least **2 agents** (this project has 8)
- [x] **LangGraph** orchestration with `StateGraph` and `Send` API
- [x] Clear **separation of responsibilities** between agents
- [x] Functional **Streamlit** interface with live progress updates
- [x] Structured code following the provided folder layout
- [x] Code documentation — all modules, functions, and key logic blocks are commented