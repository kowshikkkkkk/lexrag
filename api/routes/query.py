from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from api.schemas import QueryRequest, QueryResponse, Source
from config.exceptions import BelowConfidenceThresholdError
from config.constants import INSUFFICIENT_INFO_RESPONSE
from retrieval.retriever import retriever
from retrieval.reranker import reranker
from generation.query_rewriter import query_rewriter
from generation.generator import generator
from observability.logger import setup_logger, Timer

logger = setup_logger(__name__)
router = APIRouter(prefix="/query", tags=["Query"])


def _run_pipeline(request: QueryRequest) -> tuple[str, str, list, dict]:
    """
    Shared pipeline logic for both standard and streaming endpoints.
    Returns: original_query, rewritten_query, reranked_chunks, generation_result
    """
    original_query = request.query

    # Step 1 — Query rewriting
    if request.rewrite:
        rewritten = query_rewriter.rewrite(original_query)
    else:
        rewritten = original_query

    # Step 2 — Hybrid retrieval
    filters = {"doc_type": request.doc_type} if request.doc_type else None
    chunks = retriever.retrieve(rewritten, top_k=request.top_k, filters=filters)

    # Step 3 — Reranking
    reranked = reranker.rerank(rewritten, chunks, top_k=request.top_k)

    return original_query, rewritten, reranked


@router.post("", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Ask a question grounded in ingested legal documents.
    Returns a cited answer with source references.
    """
    with Timer("full_query_pipeline", logger) as t:
        original_query, rewritten, reranked = _run_pipeline(request)

        # Step 4 — Generate
        result = generator.generate(rewritten, reranked)

    sources = [Source(**s) for s in result["sources"]]

    logger.info(
        "Query pipeline complete",
        extra={"query": original_query[:80], "latency_ms": round(t.elapsed_ms, 2)}
    )

    return QueryResponse(
        query=original_query,
        rewritten_query=rewritten,
        answer=result["answer"],
        sources=sources,
        model=result["model"],
    )


@router.get("/stream")
async def stream_query(
    query: str,
    doc_type: str = None,
    rewrite: bool = True,
):
    """
    Streaming version of query — returns tokens via SSE as they are generated.
    """
    request = QueryRequest(query=query, doc_type=doc_type, rewrite=rewrite)
    original_query, rewritten, reranked = _run_pipeline(request)

    def event_stream():
        try:
            for token in generator.stream(rewritten, reranked):
                # SSE format — each event is "data: <content>\n\n"
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
            "X-Accel-Buffering": "no",  # disable nginx buffering
        }
    )