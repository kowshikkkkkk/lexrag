from sentence_transformers import CrossEncoder
from typing import Optional

from config.settings import get_settings
from config.exceptions import RetrievalError
from observability.logger import setup_logger, Timer

logger = setup_logger(__name__)
settings = get_settings()


class Reranker:
    """
    Cross-encoder reranker.
    Takes query + list of chunks, returns reranked list.
    """
    _instance: Optional["Reranker"] = None
    _model: Optional[CrossEncoder] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def load(self):
        if self._model is not None:
            return
        logger.info(
            "Loading reranker model",
            extra={"model": settings.rerank_model}
        )
        try:
            self._model = CrossEncoder(settings.rerank_model)
            logger.info("Reranker model loaded successfully")
        except Exception as e:
            raise RetrievalError(f"Failed to load reranker: {e}")

    def rerank(self, query: str, chunks: list[dict], top_k: int = None) -> list[dict]:
        """
        Rerank chunks using cross-encoder scores.
        Returns top_k chunks sorted by reranker score.
        """
        if self._model is None:
            raise RetrievalError("Reranker not loaded. Call reranker.load() on startup.")

        if not chunks:
            return []

        top_k = top_k or settings.final_top_k

        # Build query-chunk pairs for cross-encoder
        pairs = [[query, chunk["text"]] for chunk in chunks]

        with Timer("reranking", logger) as t:
            scores = self._model.predict(pairs)

        # Attach reranker scores
        for chunk, score in zip(chunks, scores):
            chunk["rerank_score"] = round(float(score), 4)

        # Sort by reranker score descending
        reranked = sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)
        result = reranked[:top_k]

        logger.info(
            "Reranking complete",
            extra={
                "input_chunks": len(chunks),
                "output_chunks": len(result),
                "top_score": result[0]["rerank_score"] if result else 0,
                "latency_ms": round(t.elapsed_ms, 2),
            }
        )
        return result


# Module-level singleton
reranker = Reranker()