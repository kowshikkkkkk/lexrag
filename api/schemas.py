from pydantic import BaseModel, Field
from typing import Optional


# ── Ingest ────────────────────────────────────────────────────────────────────

class IngestResponse(BaseModel):
    source: str
    doc_type: str
    chunks_ingested: int
    file_hash: str
    message: str = "Document ingested successfully"


# ── Query ─────────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=1000)
    doc_type: Optional[str] = None      # optional metadata filter
    rewrite: bool = True                 # toggle query rewriting
    top_k: Optional[int] = Field(None, ge=1, le=20)


class Source(BaseModel):
    document: str
    section: Optional[str] = None
    doc_type: Optional[str] = None
    rerank_score: Optional[float] = None


class QueryResponse(BaseModel):
    query: str
    rewritten_query: Optional[str] = None
    answer: str
    sources: list[Source]
    model: str


# ── Health ────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    service: str
    version: str


# ── Human Review ──────────────────────────────────────────────────────────────

class ReviewItem(BaseModel):
    review_id: str
    query: str
    rewritten_query: str
    retrieved_chunks: list[dict]
    draft_answer: str
    sources: list[Source]


class ReviewDecision(BaseModel):
    review_id: str
    approved: bool
    corrected_answer: Optional[str] = None
    reviewer_note: Optional[str] = None