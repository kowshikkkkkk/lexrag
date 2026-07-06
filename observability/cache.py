import hashlib
import json
from typing import Optional

import numpy as np
import redis

from config.settings import get_settings
from observability.logger import setup_logger

logger = setup_logger(__name__)
settings = get_settings()

# Redis SET holding the keys of all currently-cached entries that have an
# embedding attached. This is what the semantic layer scans instead of
# running KEYS/SCAN over the whole keyspace on every lookup.
SEMANTIC_INDEX_KEY = "lexrag:cache:index"


class QueryCache:
    """
    Two-layer Redis query cache.

    Layer 1 — exact match: O(1) hash lookup on the normalized query string.
    Fast path for literal repeat queries.

    Layer 2 — semantic match: on an exact-match miss, embeds the query and
    compares it (cosine similarity, via dot product since embeddings are
    pre-normalized) against embeddings of recently cached queries. Catches
    paraphrases like "punishment for theft" vs "penalty for stealing" that
    the exact-match layer would treat as unrelated.

    Design trade-off, worth being upfront about: the semantic layer is a
    brute-force scan over a bounded candidate set (semantic_cache_max_candidates),
    not an ANN index. At cache scale — hundreds of live entries within a TTL
    window — that scan is sub-millisecond and needs no extra infrastructure.
    If the number of concurrently-cached distinct queries ever grew into the
    tens of thousands, the right next step is a proper vector index (a small
    Qdrant collection, or RediSearch VSS) instead of scanning in Python.
    """
    _instance: Optional["QueryCache"] = None
    _client: Optional[redis.Redis] = None
    _enabled: bool = True

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def connect(self):
        """Connect to Redis."""
        if not settings.redis_enabled:
            self._enabled = False
            logger.info("Redis cache disabled")
            return

        try:
            self._client = redis.Redis(
                host=settings.redis_host,
                port=settings.redis_port,
                decode_responses=True,
            )
            self._client.ping()
            logger.info(
                "Connected to Redis",
                extra={"host": settings.redis_host, "port": settings.redis_port}
            )
        except Exception as e:
            logger.warning(f"Redis connection failed — cache disabled: {e}")
            self._enabled = False

    def _make_key(self, query: str, doc_type: Optional[str], rewrite: bool) -> str:
        """Create a cache key from query parameters."""
        payload = f"{query.lower().strip()}|{doc_type or ''}|{rewrite}"
        return "lexrag:query:" + hashlib.md5(payload.encode()).hexdigest()

    # ── Embedding helper ─────────────────────────────────────────────────────

    def _embed(self, query: str) -> Optional[np.ndarray]:
        """
        Lazily imports the embedder to avoid a hard dependency for callers
        that run with semantic caching disabled.
        """
        try:
            from embeddings.embedder import embedder
            return embedder.embed_query(query)
        except Exception as e:
            logger.warning(f"Semantic cache embedding failed: {e}")
            return None

    # ── Exact-match layer ────────────────────────────────────────────────────

    def _get_exact(self, key: str) -> Optional[dict]:
        cached = self._client.get(key)
        if not cached:
            return None
        try:
            entry = json.loads(cached)
            return entry.get("result")
        except Exception:
            return None

    # ── Semantic layer ───────────────────────────────────────────────────────

    def _get_semantic(self, query: str, doc_type: Optional[str], rewrite: bool) -> Optional[dict]:
        if not settings.semantic_cache_enabled:
            return None

        query_vec = self._embed(query)
        if query_vec is None:
            return None

        try:
            candidate_keys = self._client.smembers(SEMANTIC_INDEX_KEY)
        except Exception as e:
            logger.warning(f"Semantic cache index read failed: {e}")
            return None

        if not candidate_keys:
            return None

        best_score = -1.0
        best_entry = None
        stale_keys = []

        for key in list(candidate_keys)[: settings.semantic_cache_max_candidates]:
            raw = self._client.get(key)
            if raw is None:
                # TTL already expired the entry itself — the index reference
                # is now dangling, prune it lazily
                stale_keys.append(key)
                continue

            try:
                entry = json.loads(raw)
            except Exception:
                stale_keys.append(key)
                continue

            embedding = entry.get("embedding")
            if embedding is None:
                continue

            # Only compare against entries built with the same filters —
            # a semantic match under a different doc_type filter is a
            # different query in practice
            if entry.get("doc_type", "") != (doc_type or "") or entry.get("rewrite") != rewrite:
                continue

            cached_vec = np.array(embedding, dtype=np.float32)
            score = float(np.dot(query_vec, cached_vec))
            if score > best_score:
                best_score = score
                best_entry = entry

        if stale_keys:
            try:
                self._client.srem(SEMANTIC_INDEX_KEY, *stale_keys)
            except Exception:
                pass

        if best_entry is not None and best_score >= settings.semantic_cache_threshold:
            logger.info(
                "Cache HIT (semantic)",
                extra={
                    "query": query[:80],
                    "matched_query": best_entry.get("query", "")[:80],
                    "similarity": round(best_score, 4),
                }
            )
            return best_entry.get("result")

        return None

    # ── Public API ───────────────────────────────────────────────────────────

    def get(self, query: str, doc_type: Optional[str], rewrite: bool) -> Optional[dict]:
        """
        Get cached result. Tries exact match first, then semantic match.
        Returns None on a full miss.
        """
        if not self._enabled or self._client is None:
            return None

        key = self._make_key(query, doc_type, rewrite)

        try:
            result = self._get_exact(key)
            if result is not None:
                logger.info("Cache HIT (exact)", extra={"query": query[:80]})
                return result
        except Exception as e:
            logger.warning(f"Redis exact get failed: {e}")

        try:
            result = self._get_semantic(query, doc_type, rewrite)
            if result is not None:
                return result
        except Exception as e:
            logger.warning(f"Semantic cache lookup failed: {e}")

        logger.debug("Cache MISS", extra={"query": query[:80]})
        return None

    def set(self, query: str, doc_type: Optional[str], rewrite: bool, result: dict):
        """Cache a query result, along with its embedding for semantic lookups."""
        if not self._enabled or self._client is None:
            return

        key = self._make_key(query, doc_type, rewrite)

        embedding = None
        if settings.semantic_cache_enabled:
            vec = self._embed(query)
            if vec is not None:
                embedding = vec.tolist()

        entry = {
            "query": query,
            "doc_type": doc_type or "",
            "rewrite": rewrite,
            "result": result,
        }
        if embedding is not None:
            entry["embedding"] = embedding

        try:
            self._client.setex(key, settings.redis_ttl, json.dumps(entry))

            if embedding is not None:
                self._client.sadd(SEMANTIC_INDEX_KEY, key)
                # Keep the index set itself bounded to roughly the same
                # lifetime as its entries, so it can't grow forever if
                # entries expire without ever being scanned again
                self._client.expire(SEMANTIC_INDEX_KEY, settings.redis_ttl * 2)

            logger.info(
                "Cache SET",
                extra={
                    "query": query[:80],
                    "ttl": settings.redis_ttl,
                    "semantic_indexed": embedding is not None,
                }
            )
        except Exception as e:
            logger.warning(f"Redis set failed: {e}")

    def invalidate(self, pattern: str = "lexrag:query:*"):
        """Clear all cached queries. Called after new document ingestion."""
        if not self._enabled or self._client is None:
            return

        try:
            keys = self._client.keys(pattern)
            if keys:
                self._client.delete(*keys)
                logger.info("Cache invalidated", extra={"keys_deleted": len(keys)})
            self._client.delete(SEMANTIC_INDEX_KEY)
        except Exception as e:
            logger.warning(f"Cache invalidation failed: {e}")

    @property
    def is_enabled(self) -> bool:
        return self._enabled


# Module-level singleton
query_cache = QueryCache()