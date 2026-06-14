# ⚖️ LexRAG — Production-Grade Legal RAG System

A production-ready Retrieval-Augmented Generation system for Indian legal documents. LexRAG answers questions grounded strictly in ingested legal texts — IPC, Constitution, Supreme Court judgments, Contract Act — with source citations, confidence gating, and human-in-the-loop review.

---

## 🏗️ System Architecture
User Query

│

▼

┌─────────────────┐

│  Query Rewriter  │  ── Groq (llama-3.1-8b-instant)

│  (LLM-based)    │  ── Expands abbreviations, removes section hallucination

└────────┬────────┘

│

▼

┌─────────────────────────────────────────┐

│           Hybrid Retrieval              │

│                                         │

│  ┌──────────────┐  ┌─────────────────┐  │

│  │ Dense Search │  │  Sparse Search  │  │

│  │  (Qdrant)   │  │    (BM25)       │  │

│  │  top-20     │  │    top-20       │  │

│  └──────┬───────┘  └───────┬─────────┘  │

│         │                  │            │

│         └────────┬─────────┘            │

│                  ▼                      │

│           RRF Fusion (k=60)             │

└──────────────────┬──────────────────────┘

│

▼

┌─────────────────────────────────────────┐

│         Cross-Encoder Reranking         │

│   ms-marco-MiniLM-L-6-v2               │

│   top-20 → top-5                        │

└──────────────────┬──────────────────────┘

│

▼

┌─────────────────────────────────────────┐

│         Confidence Gate                 │

│                                         │

│  score < 0.30  → Insufficient Info     │

│  score < 2.0   → Human Review Queue    │

│  score >= 2.0  → Direct Answer         │

└──────────────────┬──────────────────────┘

│

▼

┌─────────────────────────────────────────┐

│         Generation (Groq)               │

│   llama-3.3-70b-versatile              │

│   Token budget: 3000 tokens            │

│   Citation enforcement                  │

│   SSE Streaming                         │

└──────────────────┬──────────────────────┘

│

▼

Cited Answer

---

## 🎯 Key Design Decisions

### 1. Hybrid Retrieval over Dense-Only
Dense vector search finds semantically similar chunks but fails on exact matches — if a user types "Section 420" verbatim, dense search might miss it. BM25 handles exact keyword matches perfectly. RRF fusion combines both ranked lists giving a chunk that ranks high in both a significantly higher combined score. This is the biggest retrieval quality improvement over a naive RAG system.

### 2. Section-Aware Chunking for Legal Documents
Legal Acts have explicit structure: Parts → Chapters → Sections. Naive character-based chunking splits "Section 420 defines cheating as..." mid-sentence across two chunks. Our chunker detects section headers via regex and splits at section boundaries. Each chunk stays within one section and carries `section_number` metadata for precise citation.

We chose recursive splitting over semantic chunking because:
- Legal documents have explicit structure more reliable than cosine similarity drops
- Semantic chunking requires embedding every sentence before chunking — expensive
- The article's maturity model recommends: recursive → eval stable → then semantic

### 3. Cross-Encoder Reranking as Second Stage
Bi-encoder embeddings (used in dense retrieval) encode query and document independently. Cross-encoders process the query-document pair together, giving much more accurate relevance scores. Running cross-encoder on the full corpus would be too slow — we run it only on the top-20 fused results. The score gap between relevant and irrelevant chunks is dramatic (9.5 vs 1.3 in testing).

### 4. Confidence Gating prevents Hallucination
Legal is a high-stakes domain. Instead of generating an answer when retrieval quality is low, we have two gates:
- Below `MIN_SIMILARITY_THRESHOLD (0.30)` → return fixed "insufficient information" string
- Below `REVIEW_THRESHOLD (2.0)` → send to human review queue with draft answer

This means the system never confidently answers when it doesn't have reliable grounding.

### 5. Query Rewriting without Section Hallucination
Early testing showed the LLM rewriter adding section numbers it assumed from training knowledge ("Section 415 for cheating") even when the document only had Section 420. This caused retrieval misses. The fix was explicit prompt instructions: never add section numbers unless the user mentioned one. Faithfulness score improved from 0.667 to 0.889 after this fix.

### 6. Human Review Queue builds Golden Dataset Organically
Instead of manually writing QA pairs for evaluation, every approved review item is saved as a golden dataset entry — question, rewritten query, answer, retrieved contexts, sources. Real user queries with human-verified answers are better ground truth than synthetic data.

### 7. Singleton Pattern for Heavy Models
The embedding model (~400MB) and reranker are loaded once on startup using the singleton pattern. Every import across the codebase gets the same already-loaded instance. No repeated file reads, no per-request model loading.

### 8. Structured JSON Logging with Trace IDs
Every log line is a JSON object with timestamp, level, module, and a trace ID set per request in FastAPI middleware. The same trace ID appears in every log line across ingestion, retrieval, reranking, and generation for a single request. This makes debugging production issues trivial.

---

## 🛠️ Tech Stack

| Component | Tool | Why |
|---|---|---|
| LLM | Groq (llama-3.3-70b) | Fastest inference, free tier |
| Embeddings | BAAI/bge-base-en-v1.5 | Top MTEB score, 768 dims, free |
| Vector DB | Qdrant | Production-grade, Docker-native |
| Sparse Search | BM25 (rank-bm25) | Exact keyword matching |
| Reranker | ms-marco-MiniLM-L-6-v2 | Fast cross-encoder, strong performance |
| API | FastAPI + Pydantic | Async, type-safe, auto docs |
| Streaming | SSE (Server-Sent Events) | Real-time token streaming |
| Tracking | MLflow | Experiment comparison across configs |
| Evaluation | Custom LLM-judge harness | RAGAS-compatible metrics |
| Dashboard | Streamlit | Rapid UI, Python-native |
| CI/CD | GitHub Actions | Automated testing on every push |
| Containers | Docker + docker-compose | Reproducible deployment |
| Protocol | MCP (Model Context Protocol) | Agent-accessible tools |

---

## 📁 Project Structure
lexrag/

├── config/

│   ├── settings.py          # Pydantic settings — single source of truth

│   ├── constants.py         # No magic numbers — system prompt, RRF_K etc

│   └── exceptions.py        # Typed exception hierarchy per layer

├── ingestion/

│   ├── loader.py            # PDF/TXT/DOCX loading, normalization, file hash

│   └── pipeline.py          # Full ingest pipeline: load→chunk→embed→store

├── chunking/

│   └── splitter.py          # Recursive + section-aware chunking

├── embeddings/

│   └── embedder.py          # Singleton BGE embedder with batching + retry

├── vectorstore/

│   └── store.py             # Qdrant wrapper — upsert, search, dedup

├── retrieval/

│   ├── retriever.py         # Hybrid retrieval: dense + BM25 + RRF fusion

│   └── reranker.py          # Cross-encoder reranking

├── generation/

│   ├── query_rewriter.py    # LLM-based query rewriting

│   └── generator.py         # Generation with token budget + SSE streaming

├── api/

│   ├── app.py               # FastAPI factory, middleware, exception handlers

│   ├── schemas.py           # Pydantic request/response schemas

│   ├── mcp_server.py        # MCP server exposing RAG tools to agents

│   └── routes/

│       ├── ingest.py        # POST /ingest

│       ├── query.py         # POST /query, GET /query/stream

│       └── review.py        # GET /review/pending, POST /review/decide

├── evaluation/

│   └── ragas_eval.py        # LLM-judge evaluation: faithfulness, relevancy etc

├── observability/

│   ├── logger.py            # JSON structured logging with trace IDs

│   └── mlflow_tracker.py    # MLflow experiment tracking

├── static/

│   └── dashboard.py         # Streamlit dashboard

├── tests/

│   └── unit/

│       ├── test_loader.py

│       └── test_chunker.py

├── docker/

│   ├── Dockerfile

│   └── docker-compose.yml

└── .github/

└── workflows/

└── ci.yml
---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Docker Desktop
- Groq API key (free at console.groq.com)

### 1. Clone and setup

```bash
git clone https://github.com/kowshikkkkkk/lexrag.git
cd lexrag
python -m venv venv
source venv/Scripts/activate  # Windows Git Bash
pip install -r requirements.txt
cp .env.example .env
# Add your GROQ_API_KEY to .env
```

### 2. Start Qdrant

```bash
cd docker
docker-compose up qdrant -d
cd ..
```

### 3. Start the API

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Start the Dashboard

```bash
streamlit run static/dashboard.py
```

Open `http://localhost:8501`

### 5. API Documentation

Open `http://localhost:8000/api/v1/docs`

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/v1/ingest` | Upload and index a legal document |
| POST | `/api/v1/query` | Ask a question, get cited answer |
| GET | `/api/v1/query/stream` | Same as query but SSE streaming |
| GET | `/api/v1/review/pending` | Get pending human review items |
| POST | `/api/v1/review/decide` | Approve or correct a review item |
| GET | `/api/v1/review/approved` | Get golden dataset |
| GET | `/api/v1/health` | Service health check |

---

## 📊 Evaluation Results

Evaluated on 3 golden QA pairs from Indian Penal Code:

| Metric | Score |
|---|---|
| Faithfulness | 0.889 |
| Answer Relevancy | 0.731 |
| Context Precision | 0.667 |
| Context Recall | 0.667 |

Faithfulness improved from 0.667 → 0.889 after fixing query rewriter prompt to stop hallucinating section numbers.

---

## 🤖 MCP Integration

LexRAG exposes three tools via MCP for use by AI agents:

- `query_legal` — ask a legal question
- `ingest_document` — index a new document
- `list_ingested_documents` — list knowledge base contents

Add to Claude Desktop config:

```json
{
  "mcpServers": {
    "lexrag": {
      "command": "python",
      "args": ["path/to/lexrag/api/mcp_server.py"],
      "env": {
        "PYTHONPATH": "path/to/lexrag"
      }
    }
  }
}
```

---

## 🔬 MLflow Experiment Tracking

```bash
mlflow ui --backend-store-uri sqlite:///data/mlflow/mlflow.db --port 5000
```

Open `http://localhost:5000` to compare runs across different chunking configs, top-k values, and model choices.

---

## 🧪 Running Tests

```bash
pytest tests/unit/ -v
```

---


