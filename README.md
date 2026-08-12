## Live Demos

| Platform | URL | Notes |
| --- | --- | --- |
| Render | https://langchain-copilot.onrender.com/docs | REST API |

---

## Deploy Your Own

### Option 1: Render

1. Fork this repository
2. Go to https://render.com → New Web Service
3. Connect your fork
4. Set build command: `pip install -r requirements.txt`
5. Set start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
6. Add environment variables from `.env.example`
7. Set `CHROMA_PATH=chroma_db`

### Option 2: Docker (local)

```bash
git clone https://github.com/YOUR_USERNAME/langchain-copilot
cd langchain-copilot
cp .env.example .env
# Fill in your API keys
docker-compose up
```

## Live Demo

| URL | Description |
| --- | --- |
| https://langchain-copilot.onrender.com | Redirects to API docs |
| https://langchain-copilot.onrender.com/docs | Interactive API documentation |
| https://langchain-copilot.onrender.com/health | Service health check |

---

## title: LangChain Research Copilot emoji: 🔬 colorFrom: blue colorTo: green sdk: docker pinned: false

# 🔬 LangChain Research Copilot

An AI-powered research assistant built with LangChain, LangGraph, and FastAPI. Upload any PDF document and interact with it through natural language — ask questions, extract structured data, perform calculations, and search the web for information beyond the document.

**Live Demo:** https://langchain-copilot.onrender.com/docs\
**GitHub:** https://github.com/codextivity/langchain-copilot

---

## What The Agent Can Do

The agent is not a simple chatbot. It decides autonomously how to answer each question using four capabilities:

### 1. Answer From Your Documents

Upload any PDF and ask questions about it in natural language. The agent retrieves the most relevant sections and generates a cited answer grounded in your document.

```
You:   What was Cambodia's GDP in 2013?
Agent: Cambodia's GDP in 2013 was $15,228.0 million USD (Page 4).
```

### 2. Handle Multi-Turn Conversations

Follow-up questions work correctly. The agent rewrites ambiguous follow-ups into explicit queries before searching — so pronouns and relative references resolve correctly.

```
You:   What was Cambodia's GDP in 2013?
Agent: $15,228.0 million USD (Page 4).

You:   What about 10 years later?
Agent: ← rewrites to "What was Cambodia's GDP in 2023?"
       Cambodia's GDP in 2023 was $31,940.0 million USD (Page 4).

You:   How much did it grow?
Agent: ← rewrites to "How much did Cambodia's GDP grow from 2013 to 2023?"
       Cambodia's GDP grew by approximately 109.7% over that period.
```

### 3. Use A Calculator For Precise Arithmetic

When a question requires calculation, the agent calls a calculator tool instead of asking the LLM to do arithmetic — which is unreliable. Results are exact, not estimated.

```
You:   What is the CAGR from 2013 to 2023?
Agent: ← calls compute_growth_rate(15228, 31940, periods=10)
       The CAGR from 2013 to 2023 is 7.69% per year.

You:   If Cambodia maintains 9.43% growth, what will GDP be in 5 years?
Agent: ← calls calculator("31940 * (1.0943 ** 5)")
       Cambodia's GDP would reach approximately $50,121 million in 5 years.
```

### 4. Search The Web For Information Outside The Document

When a question cannot be answered from the uploaded documents, the agent automatically searches the web instead of saying "I don't know."

```
You:   What is the best restaurant in Phnom Penh?
Agent: ← calls web_search("best restaurants Phnom Penh 2025")
       Some of the best restaurants in Phnom Penh include Malis,
       known for fine dining and Cambodian cuisine...

You:   What is Cambodia's current population?
Agent: ← calls web_search("Cambodia population 2025")
       As of 2025, Cambodia's population is approximately 17.8 million.
```

### 5. Extract Structured Data From Documents

Instead of prose answers, extract typed and validated data you can use in spreadsheets, databases, or downstream systems.

```
POST /extract
{
  "query": "Cambodia GDP 2013",
  "extraction_target": "GDP data for 2013",
  "schema_type": "gdp_datapoint"
}

Response:
{
  "year": 2013,
  "gdp_usd_millions": 15228.0,
  "growth_rate_percent": 8.35,
  "data_source_page": 4
}
```

---

## Sample Document

The live demo comes pre-loaded with an **Economic Development in Cambodia**report covering GDP data, growth rates, and economic trends from 2000 to 2023.

Try these questions on the live demo:

```
"What was Cambodia's GDP in 2013?"
"How did Cambodia's economy grow from 2000 to 2023?"
"What caused the slowdown in 2009?"
"Calculate the CAGR from 2000 to 2023"
"What are the main drivers of Cambodia's economic growth?"
"Compare growth before and after the 2009 financial crisis"
"What would Cambodia's GDP be in 2030 at current growth rates?"
"What is Cambodia's current population?"  ← triggers web search
```

---

## API Endpoints

| Method | Endpoint | Description |
| --- | --- | --- |
| GET | `/health` | Service status and version |
| POST | `/ingest` | Upload a PDF document |
| POST | `/chat` | Conversational Q&A with the agent |
| POST | `/extract` | Extract structured data from documents |
| GET | `/documents` | List all ingested documents |

### Example: Chat Request

```bash
curl -X POST https://langchain-copilot.onrender.com/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What was Cambodia GDP in 2013?",
    "history": []
  }'
```

### Example: Multi-Turn Chat

```bash
curl -X POST https://langchain-copilot.onrender.com/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What about 2023?",
    "history": [
      {"role": "human", "content": "What was Cambodia GDP in 2013?"},
      {"role": "ai", "content": "Cambodia GDP in 2013 was $15,228 million."}
    ]
  }'
```

### Example: Structured Extraction

```bash
curl -X POST https://langchain-copilot.onrender.com/extract \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Cambodia GDP 2013 growth rate",
    "extraction_target": "GDP data for Cambodia in 2013",
    "schema_type": "gdp_datapoint"
  }'
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    FastAPI Backend                       │
│         /ingest  /chat  /extract  /documents            │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│                  LangGraph Agent                         │
│                                                         │
│  Question received                                      │
│       │                                                 │
│       ├── In documents? ──────────────► Answer directly │
│       ├── Needs calculation? ────────► Calculator tool  │
│       └── Not in documents? ────────► Web search tool   │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│                   RAG Pipeline                           │
│                                                         │
│  Ingestion:  PDF → chunks → embeddings → ChromaDB       │
│  Retrieval:  question → rewrite → retrieve → generate   │
└─────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Component | Technology | Purpose |
| --- | --- | --- |
| LLM | GPT-4o / GPT-4o-mini | Generation and reasoning |
| Embeddings | text-embedding-3-small | Semantic similarity search |
| Vector store | ChromaDB | Document chunk storage and retrieval |
| Agent orchestration | LangGraph | Stateful agent with tool routing |
| LLM framework | LangChain 1.3 | RAG chains and LCEL composition |
| API | FastAPI | Async HTTP endpoints |
| Observability | LangSmith | Full chain tracing and evaluation |
| Web search | Tavily | Real-time web search for agents |
| Deployment | Render + Docker | Cloud hosting and containerization |

---

## Evaluation Results

Evaluated on a 20-question benchmark covering factual, analytical, comparative, and boundary questions using LangSmith automated scoring:

| Metric | Score | Description |
| --- | --- | --- |
| Correctness | 1.00 | Every answer factually matches ground truth |
| Faithfulness | 0.90+ | Answers stay within retrieved document context |

---

## Quick Start

### Option 1: Use The Live Demo

Visit the interactive API documentation and try it directly in your browser:

```
https://langchain-copilot.onrender.com/docs
```

1. The Cambodia economics document is pre-loaded
2. Use `POST /chat` to ask questions
3. Use `POST /extract` to extract structured data
4. Use `POST /ingest` to upload your own PDF

### Option 2: Run Locally

```bash
# Clone
git clone https://github.com/codextivity/langchain-copilot
cd langchain-copilot

# Install
pip install -r requirements.txt

# Configure
cp .env.example .env
# Add your API keys to .env

# Run
uvicorn app.main:app --reload --port 8000

# Open
http://localhost:8000/docs
```

### Option 3: Run With Docker

```bash
git clone https://github.com/codextivity/langchain-copilot
cd langchain-copilot
cp .env.example .env
# Add your API keys to .env
docker-compose up
```

---

## Configuration

Copy `.env.example` to `.env` and fill in your keys:

```bash
# Required
OPENAI_API_KEY=your-openai-api-key

# Required for tracing (https://smith.langchain.com)
LANGCHAIN_API_KEY=your-langsmith-api-key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=langchain-copilot

# Optional — enables web search (https://tavily.com)
TAVILY_API_KEY=your-tavily-api-key

# Storage
CHROMA_PATH=chroma_db
CHUNK_SIZE=1500
CHUNK_OVERLAP=400
```

---

## Project Structure

```
langchain-copilot/
├── app/
│   ├── main.py                  # FastAPI entry point with lifespan
│   ├── config.py                # Typed settings via pydantic-settings
│   ├── api/
│   │   └── routes/
│   │       ├── health.py        # GET /health + root redirect
│   │       ├── ingest.py        # POST /ingest
│   │       ├── chat.py          # POST /chat
│   │       ├── extract.py       # POST /extract
│   │       └── documents.py     # GET /documents
│   ├── core/
│   │   ├── agent.py             # LangGraph agent with tool routing
│   │   ├── chain.py             # Conversational RAG chain
│   │   ├── ingestion.py         # Idempotent PDF ingestion pipeline
│   │   ├── tools.py             # Calculator, web search, growth rate
│   │   ├── schemas.py           # Pydantic structured output models
│   │   ├── extraction_chain.py  # Structured data extraction
│   │   └── memory.py            # Chat history management
│   └── evaluation/
│       ├── create_dataset.py    # LangSmith evaluation dataset
│       └── run_evaluation.py    # Automated scoring pipeline
├── samples/                     # Sample PDF for auto-ingestion on startup
├── Dockerfile                   # Container definition
├── docker-compose.yml           # Local development setup
├── .env.example                 # Environment variable template
└── requirements.txt             # Python dependencies
```

---

## Key Engineering Decisions

**Why history-aware retrieval**?Follow-up questions like "what about 5 years later?" produce poor vector search results because they have no semantic anchor. The history-aware retriever rewrites them into explicit standalone queries before retrieval, dramatically improving multi-turn accuracy.

**Why LangGraph over AgentExecutor**?LangGraph gives explicit control over agent state and routing. Every decision point is visible in LangSmith traces and testable in isolation. AgentExecutor is a black box with no conditional logic support.

**Why idempotent ingestion**?Uploading the same document twice with naive ingestion silently doubles every chunk in the vector store, wasting retrieval slots and inflating costs. Hash-based deduplication prevents this transparently — the same file can be uploaded any number of times.

**Why separate retrieval and generation evaluation**?Most teams evaluate only final answer quality and miss retrieval failures entirely. Evaluating faithfulness independently reveals whether the LLM is hallucinating beyond retrieved context — a different problem requiring a different fix.

---

## Deployment Notes

The live demo runs on Render free tier. On each service restart, the Cambodia economics sample document is automatically re-ingested at startup so the service is always ready without manual intervention.

For production with persistent storage:

- Render paid tier with a disk mount at `/app/chroma_db`
- Replace ChromaDB with Qdrant Cloud or Pinecone for managed persistence

---

## Version 2 Roadmap

- [ ] Hybrid search — BM25 + dense embeddings for better retrieval precision

- [ ] Cross-encoder reranking — improve chunk selection accuracy

- [ ] Streamlit frontend — visual chat interface

- [ ] PostgreSQL — persistent conversation history across sessions

- [ ] Multi-document comparison endpoint

- [ ] Role-based access control

---

## Author

Built by [Codextivity](https://github.com/codextivity) as a portfolio project demonstrating production-grade LLM application development with LangChain, LangGraph, and FastAPI.