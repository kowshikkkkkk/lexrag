from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)
import numpy as np
from typing import Optional

from config.settings import get_settings
from config.exceptions import VectorStoreError, CollectionNotFoundError
from config.constants import META_FILE_HASH
from observability.logger import setup_logger, Timer

logger = setup_logger(__name__)
settings = get_settings()


class VectorStore:
    """
    Qdrant wrapper — the only place in the app that talks to Qdrant directly.
    Everything else goes through this class.
    """
    _instance: Optional["VectorStore"] = None
    _client: Optional[QdrantClient] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def connect(self):
        """Connect to Qdrant and ensure collection exists."""
        if self._client is not None:
            return

        try:
            # Local mode — no Docker needed
            # Switch to QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
            # in Step 13 when Docker is running
            self._client = QdrantClient(path="./data/qdrant")
            logger.info("Connected to Qdrant", extra={"mode": "local"})
            self._ensure_collection()
        except Exception as e:
            raise VectorStoreError(f"Failed to connect to Qdrant: {e}")

    def _ensure_collection(self):
        """Create collection if it doesn't exist."""
        collections = [c.name for c in self._client.get_collections().collections]

        if settings.qdrant_collection_name not in collections:
            self._client.create_collection(
                collection_name=settings.qdrant_collection_name,
                vectors_config=VectorParams(
                    size=settings.qdrant_vector_size,  # 768 for bge-base
                    distance=Distance.COSINE,
                ),
            )
            logger.info(
                "Created Qdrant collection",
                extra={"collection": settings.qdrant_collection_name}
            )
        else:
            logger.info(
                "Collection already exists",
                extra={"collection": settings.qdrant_collection_name}
            )

    def _ensure_connected(self):
        if self._client is None:
            raise VectorStoreError("VectorStore not connected. Call store.connect() on startup.")

    def document_exists(self, file_hash: str) -> bool:
        """
        Check if a document with this hash is already in the store.
        Used for deduplication before ingestion.
        """
        self._ensure_connected()
        results = self._client.scroll(
            collection_name=settings.qdrant_collection_name,
            scroll_filter=Filter(
                must=[FieldCondition(
                    key=META_FILE_HASH,
                    match=MatchValue(value=file_hash)
                )]
            ),
            limit=1,
        )
        return len(results[0]) > 0

    def upsert(self, chunk_ids: list[str], vectors: np.ndarray, metadata: list[dict]):
        """
        Insert or update chunks in Qdrant.
        chunk_ids are deterministic (filehash_index) so re-ingesting same file
        overwrites existing points instead of creating duplicates.
        """
        self._ensure_connected()

        points = []
        for i, (chunk_id, vector, meta) in enumerate(zip(chunk_ids, vectors, metadata)):
            # Qdrant needs integer or UUID point IDs
            # We use a hash of chunk_id to get a stable integer
            point_id = abs(hash(chunk_id)) % (2**63)
            points.append(
                PointStruct(
                    id=point_id,
                    vector=vector.tolist(),
                    payload=meta,
                )
            )

        with Timer("qdrant_upsert", logger) as t:
            self._client.upsert(
                collection_name=settings.qdrant_collection_name,
                points=points,
            )

        logger.info(
            "Upserted chunks to Qdrant",
            extra={"count": len(points), "latency_ms": round(t.elapsed_ms, 2)}
        )

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 20,
        filters: Optional[dict] = None,
    ) -> list[dict]:
        """
        Dense vector search.
        Returns list of dicts with text, metadata, and score.
        """
        self._ensure_connected()

        from qdrant_client.models import QueryRequest

        qdrant_filter = None
        if filters:
            conditions = [
                FieldCondition(key=k, match=MatchValue(value=v))
                for k, v in filters.items()
            ]
            qdrant_filter = Filter(must=conditions)

        with Timer("qdrant_search", logger) as t:
            results = self._client.query_points(
                collection_name=settings.qdrant_collection_name,
                query=query_vector.tolist(),
                limit=top_k,
                query_filter=qdrant_filter,
                with_payload=True,
            )

        hits = []
        for r in results.points:
            hits.append({
                "text": r.payload.get("text", ""),
                "metadata": r.payload,
                "score": r.score,
            })

        logger.debug(
            "Dense search complete",
            extra={"top_k": top_k, "hits": len(hits), "latency_ms": round(t.elapsed_ms, 2)}
        )
        return hits


# Module-level singleton
vector_store = VectorStore()