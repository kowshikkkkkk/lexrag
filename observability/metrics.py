from prometheus_client import Counter, Histogram, Gauge, Summary
from observability.logger import setup_logger

logger = setup_logger(__name__)

# ── Query metrics ─────────────────────────────────────────────────────────────
QUERY_COUNTER = Counter(
    "lexrag_queries_total",
    "Total number of queries processed",
    ["status"],  # success, error, review, cache_hit
)

QUERY_LATENCY = Histogram(
    "lexrag_query_latency_seconds",
    "Query pipeline latency in seconds",
    buckets=[0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0],
)

RETRIEVAL_LATENCY = Histogram(
    "lexrag_retrieval_latency_seconds",
    "Retrieval latency in seconds",
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.0],
)

RERANK_LATENCY = Histogram(
    "lexrag_rerank_latency_seconds",
    "Reranking latency in seconds",
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0],
)

GENERATION_LATENCY = Histogram(
    "lexrag_generation_latency_seconds",
    "LLM generation latency in seconds",
    buckets=[0.1, 0.25, 0.5, 1.0, 2.0, 5.0],
)

# ── Cache metrics ─────────────────────────────────────────────────────────────
CACHE_HITS = Counter(
    "lexrag_cache_hits_total",
    "Total Redis cache hits",
)

CACHE_MISSES = Counter(
    "lexrag_cache_misses_total",
    "Total Redis cache misses",
)

# ── Ingestion metrics ─────────────────────────────────────────────────────────
INGEST_COUNTER = Counter(
    "lexrag_documents_ingested_total",
    "Total documents ingested",
    ["doc_type"],
)

CHUNKS_GAUGE = Gauge(
    "lexrag_chunks_total",
    "Total chunks currently in vector store",
)

# ── Error metrics ─────────────────────────────────────────────────────────────
ERROR_COUNTER = Counter(
    "lexrag_errors_total",
    "Total errors by type",
    ["error_type"],
)

# ── Review metrics ────────────────────────────────────────────────────────────
REVIEW_COUNTER = Counter(
    "lexrag_reviews_total",
    "Total queries sent to human review",
)


def update_chunks_gauge(count: int):
    """Update the chunks gauge after ingestion."""
    CHUNKS_GAUGE.set(count)