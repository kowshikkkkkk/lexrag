import mlflow
from config.settings import get_settings
from observability.logger import setup_logger

logger = setup_logger(__name__)
settings = get_settings()


def _get_client():
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)


def init_mlflow():
    """Initialize MLflow experiment. Call once on startup."""
    _get_client()
    logger.info("MLflow initialized", extra={"experiment": settings.mlflow_experiment_name})


def log_query(
    query: str,
    rewritten_query: str,
    retrieval_latency_ms: float,
    rerank_latency_ms: float,
    generation_latency_ms: float,
    total_latency_ms: float,
    top_rerank_score: float,
    chunks_retrieved: int,
    answer_length: int,
    went_to_review: bool,
):
    """Log a complete query run to MLflow."""
    try:
        _get_client()
        with mlflow.start_run(run_name="query_run"):
            mlflow.log_params({
                "chunk_size": str(settings.chunk_size),
                "embedding_model": settings.embedding_model,
                "rerank_model": settings.rerank_model,
                "dense_top_k": str(settings.dense_top_k),
                "final_top_k": str(settings.final_top_k),
                "llm_model": settings.groq_model_quality,
            })
            mlflow.set_tags({
                "query": query[:100],
                "rewritten_query": rewritten_query[:100],
            })
            mlflow.log_metrics({
                "retrieval_latency_ms": retrieval_latency_ms,
                "rerank_latency_ms": rerank_latency_ms,
                "generation_latency_ms": generation_latency_ms,
                "total_latency_ms": total_latency_ms,
                "top_rerank_score": top_rerank_score,
                "chunks_retrieved": float(chunks_retrieved),
                "answer_length": float(answer_length),
                "went_to_review": float(went_to_review),
            })
        logger.debug("MLflow run logged")
    except Exception as e:
        logger.warning(f"MLflow logging failed silently: {e}")


def log_eval_metrics(metrics: dict):
    """Log RAGAS evaluation metrics."""
    try:
        _get_client()
        with mlflow.start_run(run_name="eval_run"):
            mlflow.log_params({
                "chunk_size": str(settings.chunk_size),
                "embedding_model": settings.embedding_model,
                "rerank_model": settings.rerank_model,
            })
            mlflow.log_metrics(metrics)
        logger.info("Eval metrics logged", extra={"metrics": metrics})
    except Exception as e:
        logger.warning(f"MLflow eval logging failed: {e}")