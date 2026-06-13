import time
from typing import Optional
import numpy as np
from sentence_transformers import SentenceTransformer

from config.settings import get_settings
from config.exceptions import EmbeddingError, EmbeddingModelNotLoadedError
from observability.logger import setup_logger, Timer

logger = setup_logger(__name__)
settings = get_settings()


class Embedder:
    """
    Singleton embedding model.
    Load once, reuse everywhere.
    """
    _instance: Optional["Embedder"] = None
    _model: Optional[SentenceTransformer] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def load(self):
        """Load the embedding model into memory."""
        if self._model is not None:
            return  # already loaded

        logger.info(
            "Loading embedding model",
            extra={"model": settings.embedding_model}
        )
        try:
            self._model = SentenceTransformer(
                settings.embedding_model,
                device=settings.embedding_device,
            )
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            raise EmbeddingError(f"Failed to load embedding model: {e}")

    def _ensure_loaded(self):
        if self._model is None:
            raise EmbeddingModelNotLoadedError(
                "Embedding model not loaded. Call embedder.load() on startup."
            )

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """
        Embed a list of texts in batches.
        Returns numpy array of shape (len(texts), 768).
        """
        self._ensure_loaded()

        if not texts:
            raise EmbeddingError("No texts provided for embedding.")

        all_embeddings = []
        batch_size = settings.embedding_batch_size
        total_batches = (len(texts) + batch_size - 1) // batch_size

        with Timer("embedding", logger) as t:
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                batch_num = (i // batch_size) + 1

                # Retry logic — up to 3 attempts with exponential backoff
                for attempt in range(3):
                    try:
                        embeddings = self._model.encode(
                            batch,
                            normalize_embeddings=True,  # cosine similarity ready
                            show_progress_bar=False,
                        )
                        all_embeddings.append(embeddings)
                        logger.debug(
                            "Batch embedded",
                            extra={
                                "batch": f"{batch_num}/{total_batches}",
                                "size": len(batch),
                            }
                        )
                        break  # success — exit retry loop

                    except Exception as e:
                        if attempt == 2:
                            raise EmbeddingError(
                                f"Embedding failed after 3 attempts: {e}"
                            )
                        wait = 2 ** attempt  # 1s, 2s, 4s
                        logger.warning(
                            f"Embedding attempt {attempt + 1} failed, retrying in {wait}s",
                            extra={"error": str(e)}
                        )
                        time.sleep(wait)

        result = np.vstack(all_embeddings)
        logger.info(
            "Embedding complete",
            extra={
                "total_texts": len(texts),
                "shape": str(result.shape),
                "latency_ms": round(t.elapsed_ms, 2),
            }
        )
        return result

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single query string.
        Separate method because queries may need different prefixes
        for some models (bge uses 'Represent this sentence for searching:')
        """
        self._ensure_loaded()

        # BGE models perform better with this prefix on queries
        prefixed = f"Represent this sentence for searching: {query}"

        with Timer("query_embedding", logger):
            embedding = self._model.encode(
                prefixed,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        return embedding


# Module-level singleton — import this everywhere
embedder = Embedder()