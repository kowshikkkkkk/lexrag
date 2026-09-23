import os
import pickle
import shutil
import tempfile
import threading
from pathlib import Path
from typing import Optional
from rank_bm25 import BM25Okapi

from observability.logger import setup_logger

logger = setup_logger(__name__)

BM25_INDEX_PATH = Path("./data/processed/bm25_index.pkl")
BM25_CORPUS_PATH = Path("./data/processed/bm25_corpus.pkl")


def tokenize(text: str) -> list[str]:
    return text.lower().split()


class PersistentBM25Index:
    """
    BM25 index that persists to disk.
    Rebuilt only when new documents are ingested.
    Loaded once on startup.
    """
    _instance: Optional["PersistentBM25Index"] = None
    _instance_lock = threading.Lock()
    _op_lock = threading.Lock()
    _bm25: Optional[BM25Okapi] = None
    _corpus: list[dict] = []

    def __new__(cls):
        # Check-then-create is not atomic — two threads racing at startup
        # can both see _instance as None and both build one.
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def load(self):
        """Load index from disk if it exists."""
        with self._op_lock:
            if self._bm25 is not None:
                return

            if BM25_INDEX_PATH.exists() and BM25_CORPUS_PATH.exists():
                try:
                    with open(BM25_INDEX_PATH, "rb") as f:
                        self._bm25 = pickle.load(f)
                    with open(BM25_CORPUS_PATH, "rb") as f:
                        self._corpus = pickle.load(f)
                    logger.info(
                        "BM25 index loaded from disk",
                        extra={"chunks": len(self._corpus)}
                    )
                except Exception as e:
                    logger.warning(f"Failed to load BM25 index: {e}. Will rebuild.")
                    self._bm25 = None
                    self._corpus = []
            else:
                logger.info("No BM25 index found on disk — will build on first ingest")

    @staticmethod
    def _atomic_pickle_dump(obj, path: Path):
        """
        Write via a temp file + atomic rename so a crash mid-write never
        leaves a half-written or mismatched pickle on disk.
        """
        fd, tmp_path = tempfile.mkstemp(dir=str(path.parent))
        try:
            with os.fdopen(fd, "wb") as f:
                pickle.dump(obj, f)
            shutil.move(tmp_path, path)
        except Exception:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise

    def build(self, chunks: list[dict]):
        """
        Build BM25 index from chunks and save to disk.
        Called after every document ingest.
        """
        if not chunks:
            logger.warning("No chunks provided to build BM25 index")
            return

        with self._op_lock:
            corpus = chunks
            tokenized = [tokenize(c["text"]) for c in chunks]
            bm25 = BM25Okapi(tokenized)

            BM25_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
            self._atomic_pickle_dump(bm25, BM25_INDEX_PATH)
            self._atomic_pickle_dump(corpus, BM25_CORPUS_PATH)

            # Only swap in-memory state once both files are safely on disk.
            self._bm25 = bm25
            self._corpus = corpus

        logger.info(
            "BM25 index built and saved to disk",
            extra={"chunks": len(chunks)}
        )

    def search(self, query: str, top_k: int) -> list[dict]:
        """Search BM25 index. Returns top_k results."""
        if self._bm25 is None or not self._corpus:
            logger.warning("BM25 index not built yet — returning empty results")
            return []

        import numpy as np
        query_tokens = tokenize(query)
        scores = self._bm25.get_scores(query_tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append({
                    "text": self._corpus[idx]["text"],
                    "metadata": self._corpus[idx]["metadata"],
                    "score": float(scores[idx]),
                })
        return results

    def rebuild_from_qdrant(self, qdrant_client, collection_name: str):
        """
        Rebuild BM25 index from all chunks in Qdrant.
        Called on startup if index is stale or missing.
        """
        try:
            results, _ = qdrant_client.scroll(
                collection_name=collection_name,
                with_payload=True,
                with_vectors=False,
                limit=100000,
            )
            chunks = []
            for r in results:
                text = r.payload.get("text", "")
                if text:
                    chunks.append({
                        "text": text,
                        "metadata": r.payload,
                    })

            if chunks:
                self.build(chunks)
                logger.info(
                    "BM25 index rebuilt from Qdrant",
                    extra={"chunks": len(chunks)}
                )
            else:
                logger.info("No chunks in Qdrant — BM25 index empty")

        except Exception as e:
            logger.warning(f"Failed to rebuild BM25 from Qdrant: {e}")

    @property
    def is_ready(self) -> bool:
        return self._bm25 is not None and len(self._corpus) > 0


# Module-level singleton
bm25_index = PersistentBM25Index()