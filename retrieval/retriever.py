import numpy as np
from rank_bm25 import BM25Okapi
from typing import Optional

from config.settings import get_settings
from config.constants import RRF_K
from config.exceptions import RetrievalError, BelowConfidenceThresholdError
from embeddings.embedder import embedder
from vectorstore.store import vector_store
from observability.logger import setup_logger, Timer

logger = setup_logger(__name__)
settings = get_settings()


def tokenize(text: str) -> list[str]:
    """Simple whitespace tokenizer for BM25."""
    return text.lower().split()


class HybridRetriever:
    """
    Combines dense (Qdrant) and sparse (BM25) retrieval
    using Reciprocal Rank Fusion.
    """

    def _get_all_chunks(self, filters: Optional[dict] = None) -> list[dict]:
        """
        Fetch all chunks from Qdrant to build BM25 index.
        In production with millions of chunks you'd maintain a separate
        BM25 index on disk — for now we build it per query from stored chunks.
        """
        try:
            results, _ = vector_store._client.scroll(
                collection_name=settings.qdrant_collection_name,
                with_payload=True,
                with_vectors=False,
                limit=10000,
            )
            chunks = []
            for r in results:
                text = r.payload.get("text", "")
                if text:
                    chunks.append({
                        "text": text,
                        "metadata": r.payload,
                        "id": r.id,
                    })
            return chunks
        except Exception as e:
            raise RetrievalError(f"Failed to fetch chunks for BM25: {e}")

    def _dense_search(
        self, query: str, top_k: int, filters: Optional[dict]
    ) -> list[dict]:
        """Embed query and search Qdrant."""
        query_vector = embedder.embed_query(query)
        return vector_store.search(query_vector, top_k=top_k, filters=filters)

    def _sparse_search(
        self, query: str, all_chunks: list[dict], top_k: int
    ) -> list[dict]:
        """BM25 search over all chunks."""
        corpus = [tokenize(c["text"]) for c in all_chunks]
        bm25 = BM25Okapi(corpus)
        query_tokens = tokenize(query)
        scores = bm25.get_scores(query_tokens)

        # Get top_k indices sorted by score
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # only include chunks with non-zero BM25 score
                results.append({
                    "text": all_chunks[idx]["text"],
                    "metadata": all_chunks[idx]["metadata"],
                    "score": float(scores[idx]),
                })
        return results

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
            key = result["text"][:100]  # use first 100 chars as key
            scores[key] = scores.get(key, 0) + 1 / (k + rank + 1)
            texts[key] = result

        # Score from sparse results
        for rank, result in enumerate(sparse_results):
            key = result["text"][:100]
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
            # Fetch all chunks for BM25
            all_chunks = self._get_all_chunks(filters)

            if not all_chunks:
                raise RetrievalError("No chunks found in vector store. Ingest documents first.")

            # Run dense and sparse search
            with Timer("dense_search", logger):
                dense_results = self._dense_search(query, dense_k, filters)

            with Timer("sparse_search", logger):
                sparse_results = self._sparse_search(query, all_chunks, sparse_k)

            # Fuse results
            fused = self._rrf_fusion(dense_results, sparse_results)

        # Confidence gate — top result must meet minimum threshold
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