from typing import Optional
import hashlib
from config.settings import get_settings
from config.exceptions import RetrievalError, BelowConfidenceThresholdError
from embeddings.embedder import embedder
from vectorstore.store import vector_store
from retrieval.bm25_index import bm25_index
from config.constants import RRF_K
from observability.logger import setup_logger, Timer

logger = setup_logger(__name__)
settings = get_settings()


class HybridRetriever:
    """
    Combines dense (Qdrant) and sparse (BM25) retrieval
    using Reciprocal Rank Fusion.
    """

    def _dense_search(
        self, query: str, top_k: int, filters: Optional[dict]
    ) -> list[dict]:
        """Embed query and search Qdrant."""
        query_vector = embedder.embed_query(query)
        return vector_store.search(query_vector, top_k=top_k, filters=filters)

    @staticmethod
    def _fusion_key(result: dict) -> str:
        """
        Unique key for RRF fusion. A 100-char text prefix collides whenever two
        different chunks share an opening line. Prefer a real chunk_id from
        metadata if present; otherwise hash the full text so only truly
        identical chunks collide.
        """
        chunk_id = (result.get("metadata") or {}).get("chunk_id")
        if chunk_id:
            return str(chunk_id)
        return hashlib.sha256(result["text"].encode("utf-8")).hexdigest()

    def _rrf_fusion(
        self,
        dense_results: list[dict],
        sparse_results: list[dict],
        k: int = RRF_K,
    ) -> list[dict]:
        """
        Reciprocal Rank Fusion.
        score = 1/(k + rank_dense) + 1/(k + rank_sparse)
        """
        scores: dict[str, float] = {}
        texts: dict[str, dict] = {}

        # Score from dense results
        for rank, result in enumerate(dense_results):
            key = self._fusion_key(result)  
            scores[key] = scores.get(key, 0) + 1 / (k + rank + 1)
            texts[key] = result

        # Score from sparse results
        for rank, result in enumerate(sparse_results):
            key = self._fusion_key(result)
            scores[key] = scores.get(key, 0) + 1 / (k + rank + 1)
            if key not in texts:
                texts[key] = result

        # Sort by combined RRF score
        sorted_keys = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        fused = []
        for key in sorted_keys:
            result = texts[key].copy()
            result["rrf_score"] = round(scores[key], 6)
            fused.append(result)

        return fused

    def retrieve(
        self,
        query: str,
        top_k: int = None,
        filters: Optional[dict] = None,
    ) -> list[dict]:
        """
        Main retrieval entry point.
        Returns top_k most relevant chunks after hybrid search and RRF fusion.
        """
        top_k = top_k or settings.final_top_k
        dense_k = settings.dense_top_k
        sparse_k = settings.sparse_top_k

        logger.info(
            "Starting hybrid retrieval",
            extra={"query": query[:80], "top_k": top_k}
        )

        with Timer("hybrid_retrieval", logger) as t:
            # Run dense and sparse search
            with Timer("dense_search", logger):
                dense_results = self._dense_search(query, dense_k, filters)

            with Timer("sparse_search", logger):
                sparse_results = bm25_index.search(query, sparse_k)

            # Fuse results
            fused = self._rrf_fusion(dense_results, sparse_results)

        # Confidence gate — top result must meet minimum threshold
        # NOTE: left exactly as-is for now — this is a known separate bug
        # (compares mismatched score scales) that we're fixing in the next step.
        if not fused:
            raise BelowConfidenceThresholdError(
                "No relevant chunks found for this query."
            )

        top_score = fused[0].get("score", fused[0].get("rrf_score", 0))
        if top_score < settings.min_similarity_threshold:
            raise BelowConfidenceThresholdError(
                "I do not have sufficient information in the provided documents "
                "to answer this question."
            )

        results = fused[:top_k]

        logger.info(
            "Retrieval complete",
            extra={
                "query": query[:80],
                "results": len(results),
                "top_score": fused[0].get("rrf_score", 0),
                "latency_ms": round(t.elapsed_ms, 2),
            }
        )

        return results


# Module-level singleton
retriever = HybridRetriever()