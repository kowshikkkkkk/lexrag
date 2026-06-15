import hashlib
import json
from typing import Optional

import redis

from config.settings import get_settings
from observability.logger import setup_logger

logger = setup_logger(__name__)
settings = get_settings()


class QueryCache:
    """
    Redis-based query cache.
    Caches full query pipeline results by query hash.
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

    def get(self, query: str, doc_type: Optional[str], rewrite: bool) -> Optional[dict]:
        """Get cached result. Returns None on miss."""
        if not self._enabled or self._client is None:
            return None

        try:
            key = self._make_key(query, doc_type, rewrite)
            cached = self._client.get(key)
            if cached:
                logger.info(
                    "Cache HIT",
                    extra={"query": query[:80]}
                )
                return json.loads(cached)
            logger.debug("Cache MISS", extra={"query": query[:80]})
            return None
        except Exception as e:
            logger.warning(f"Redis get failed: {e}")
            return None

    def set(self, query: str, doc_type: Optional[str], rewrite: bool, result: dict):
        """Cache a query result."""
        if not self._enabled or self._client is None:
            return

        try:
            key = self._make_key(query, doc_type, rewrite)
            self._client.setex(
                key,
                settings.redis_ttl,
                json.dumps(result),
            )
            logger.info(
                "Cache SET",
                extra={"query": query[:80], "ttl": settings.redis_ttl}
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
                logger.info(f"Cache invalidated", extra={"keys_deleted": len(keys)})
        except Exception as e:
            logger.warning(f"Cache invalidation failed: {e}")

    @property
    def is_enabled(self) -> bool:
        return self._enabled


# Module-level singleton
query_cache = QueryCache()