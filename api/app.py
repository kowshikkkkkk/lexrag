import uuid
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from config.settings import get_settings
from config.constants import API_PREFIX
from config.exceptions import (
    LexRAGError,
    BelowConfidenceThresholdError,
    UnsupportedFileTypeError,
    DuplicateDocumentError,
    LLMConnectionError,
)
from observability.logger import setup_logger, trace_id_var

settings = get_settings()
logger = setup_logger(__name__, settings.log_level)


def create_app() -> FastAPI:
    app = FastAPI(
        title="LexRAG",
        description="Production-grade Legal Document Retrieval and Q&A System",
        version="1.0.0",
        docs_url=f"{API_PREFIX}/docs",
        redoc_url=f"{API_PREFIX}/redoc",
    )

    @app.middleware("http")
    async def trace_middleware(request: Request, call_next):
        trace_id = request.headers.get("X-Trace-ID", str(uuid.uuid4())[:8])
        trace_id_var.set(trace_id)
        response = await call_next(request)
        response.headers["X-Trace-ID"] = trace_id
        return response

    @app.exception_handler(BelowConfidenceThresholdError)
    async def confidence_handler(request: Request, exc: BelowConfidenceThresholdError):
        return JSONResponse(
            status_code=200,
            content={"answer": str(exc), "sources": [], "confidence": "low"},
        )

    @app.exception_handler(UnsupportedFileTypeError)
    async def unsupported_file_handler(request: Request, exc: UnsupportedFileTypeError):
        return JSONResponse(
            status_code=400,
            content={"error": "UnsupportedFileType", "detail": str(exc)},
        )

    @app.exception_handler(DuplicateDocumentError)
    async def duplicate_handler(request: Request, exc: DuplicateDocumentError):
        return JSONResponse(
            status_code=409,
            content={"error": "DuplicateDocument", "detail": str(exc)},
        )

    @app.exception_handler(LLMConnectionError)
    async def llm_connection_handler(request: Request, exc: LLMConnectionError):
        return JSONResponse(
            status_code=503,
            content={"error": "LLMUnavailable", "detail": str(exc)},
        )

    @app.exception_handler(LexRAGError)
    async def generic_lexrag_handler(request: Request, exc: LexRAGError):
        logger.error(str(exc), extra={"error_type": type(exc).__name__})
        return JSONResponse(
            status_code=500,
            content={"error": type(exc).__name__, "detail": str(exc)},
        )

    @app.on_event("startup")
    async def startup():
        from embeddings.embedder import embedder
        from vectorstore.store import vector_store
        from retrieval.reranker import reranker
        from retrieval.bm25_index import bm25_index
        from observability.mlflow_tracker import init_mlflow
        from observability.cache import query_cache

        embedder.load()
        vector_store.connect()
        reranker.load()
        query_cache.connect()

        bm25_index.load()
        if not bm25_index.is_ready:
            bm25_index.rebuild_from_qdrant(
                vector_store._client,
                settings.qdrant_collection_name
            )

        init_mlflow()
        logger.info("All components initialized")

    @app.get(f"{API_PREFIX}/health")
    async def health():
        return {"status": "ok", "service": "LexRAG", "version": "1.0.0"}

    @app.get(f"{API_PREFIX}/mlflow-test")
    async def mlflow_test():
        import mlflow
        from config.settings import get_settings
        s = get_settings()
        try:
            mlflow.set_tracking_uri(s.mlflow_tracking_uri)
            mlflow.set_experiment(s.mlflow_experiment_name)
            with mlflow.start_run(run_name="api_test"):
                mlflow.log_metric("api_test_metric", 99.0)
            return {"status": "run logged"}
        except Exception as e:
            return {"error": str(e)}

    from api.routes.ingest import router as ingest_router
    from api.routes.query import router as query_router
    from api.routes.review import router as review_router
    app.include_router(ingest_router, prefix=API_PREFIX)
    app.include_router(query_router, prefix=API_PREFIX)
    app.include_router(review_router, prefix=API_PREFIX)

    return app