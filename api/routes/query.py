import time
from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from api.schemas import QueryRequest, QueryResponse, Source
from config.settings import get_settings
from config.exceptions import BelowConfidenceThresholdError
from config.constants import INSUFFICIENT_INFO_RESPONSE
from retrieval.retriever import retriever
from retrieval.reranker import reranker
from generation.query_rewriter import query_rewriter
from generation.generator import generator
from observability.logger import setup_logger, Timer
from observability.mlflow_tracker import log_query
from observability.cache import query_cache
from observability.metrics import (
    QUERY_COUNTER,
    QUERY_LATENCY,
    RETRIEVAL_LATENCY,
    RERANK_LATENCY,
    GENERATION_LATENCY,
    CACHE_HITS,
    CACHE_MISSES,
    REVIEW_COUNTER,
    ERROR_COUNTER,
)

logger = setup_logger(__name__)
settings = get_settings()
router = APIRouter(prefix="/query", tags=["Query"])


import asyncio
from concurrent.futures import ThreadPoolExecutor

executor = ThreadPoolExecutor(max_workers=4)

async def _run_pipeline_async(request: QueryRequest):
    """
    Async pipeline — runs CPU-bound steps in thread pool
    so they don't block the event loop.
    """
    original_query = request.query
    loop = asyncio.get_event_loop()

    # Step 1 — Query rewriting (I/O bound — Groq API call)
    t0 = time.perf_counter()
    if request.rewrite:
        rewritten = await loop.run_in_executor(
            executor, query_rewriter.rewrite, original_query
        )
    else:
        rewritten = original_query
    rewrite_ms = (time.perf_counter() - t0) * 1000

    # Step 2 — Hybrid retrieval (CPU bound — embedding + BM25)
    filters = {"doc_type": request.doc_type} if request.doc_type else None
    t0 = time.perf_counter()
    chunks = await loop.run_in_executor(
        executor,
        lambda: retriever.retrieve(rewritten, top_k=request.top_k, filters=filters)
    )
    retrieval_ms = (time.perf_counter() - t0) * 1000

    # Step 3 — Reranking (CPU bound — cross-encoder)
    t0 = time.perf_counter()
    reranked = await loop.run_in_executor(
        executor,
        lambda: reranker.rerank(rewritten, chunks, top_k=request.top_k)
    )
    rerank_ms = (time.perf_counter() - t0) * 1000

    return original_query, rewritten, reranked, {
        "rewrite_ms": rewrite_ms,
        "retrieval_ms": retrieval_ms,
        "rerank_ms": rerank_ms,
    }


@router.post("", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Ask a question grounded in ingested legal documents.
    Results are cached in Redis for 1 hour.
    Low confidence responses go to human review queue.
    """
    from api.routes.review import add_to_review_queue

    # Check cache first
    cached = query_cache.get(request.query, request.doc_type, request.rewrite)
    if cached:
        CACHE_HITS.inc()
        QUERY_COUNTER.labels(status="cache_hit").inc()
        return QueryResponse(**cached)

    CACHE_MISSES.inc()

    total_start = time.perf_counter()

    try:
        original_query, rewritten, reranked, timings = await _run_pipeline_async(request)
    except Exception as e:
        ERROR_COUNTER.labels(error_type=type(e).__name__).inc()
        raise

    top_score = reranked[0].get("rerank_score", 0) if reranked else 0
    needs_review = top_score < settings.review_threshold

    t0 = time.perf_counter()
    result = generator.generate(rewritten, reranked)
    generation_ms = (time.perf_counter() - t0) * 1000
    total_ms = (time.perf_counter() - total_start) * 1000

    sources = [Source(**s) for s in result["sources"]]

    # Track latency metrics
    RETRIEVAL_LATENCY.observe(timings["retrieval_ms"] / 1000)
    RERANK_LATENCY.observe(timings["rerank_ms"] / 1000)
    GENERATION_LATENCY.observe(generation_ms / 1000)
    QUERY_LATENCY.observe(total_ms / 1000)

    # Log to MLflow
    log_query(
        query=original_query,
        rewritten_query=rewritten,
        retrieval_latency_ms=timings["retrieval_ms"],
        rerank_latency_ms=timings["rerank_ms"],
        generation_latency_ms=generation_ms,
        total_latency_ms=total_ms,
        top_rerank_score=top_score,
        chunks_retrieved=len(reranked),
        answer_length=len(result["answer"]),
        went_to_review=needs_review,
    )

    if needs_review:
        REVIEW_COUNTER.inc()
        QUERY_COUNTER.labels(status="review").inc()
        review_id = add_to_review_queue(
            query=original_query,
            rewritten_query=rewritten,
            chunks=reranked,
            draft_answer=result["answer"],
            sources=result["sources"],
        )
        logger.info(
            "Low confidence — sent to review",
            extra={"review_id": review_id, "top_score": top_score}
        )
        return QueryResponse(
            query=original_query,
            rewritten_query=rewritten,
            answer=f"[Under Review: {review_id}] {result['answer']}",
            sources=sources,
            model=result["model"],
        )

    QUERY_COUNTER.labels(status="success").inc()

    response_data = QueryResponse(
        query=original_query,
        rewritten_query=rewritten,
        answer=result["answer"],
        sources=sources,
        model=result["model"],
    )

    # Cache the result
    query_cache.set(
        request.query,
        request.doc_type,
        request.rewrite,
        response_data.model_dump(),
    )

    logger.info(
        "Query pipeline complete",
        extra={"query": original_query[:80], "total_ms": round(total_ms, 2)}
    )

    return response_data


@router.get("/stream")
async def stream_query(
    query: str,
    doc_type: str = None,
    rewrite: bool = True,
):
    """Streaming version — returns tokens via SSE."""
    request = QueryRequest(query=query, doc_type=doc_type, rewrite=rewrite)
    original_query, rewritten, reranked, _ = await _run_pipeline_async(request)

    def event_stream():
        try:
            for token in generator.stream(rewritten, reranked):
                yield f"data: {token}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield f"data: [ERROR] {str(e)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        }
    )