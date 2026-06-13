from groq import Groq

from config.settings import get_settings
from config.exceptions import GenerationError, LLMConnectionError
from observability.logger import setup_logger, Timer

logger = setup_logger(__name__)
settings = get_settings()

REWRITE_PROMPT = """You are a legal query optimizer for Indian law.
Rewrite the user's question into a clear, specific query optimized for searching Indian legal documents.

Rules:
- Expand abbreviations (420 → Section 420 IPC, CrPC → Code of Criminal Procedure)
- Add legal context if missing
- Keep it concise — one sentence only
- Return ONLY the rewritten query, nothing else

User question: {query}
Rewritten query:"""


class QueryRewriter:
    """
    Rewrites user queries using Groq for better retrieval.
    """

    def __init__(self):
        try:
            self._client = Groq(api_key=settings.groq_api_key)
        except Exception as e:
            raise LLMConnectionError(f"Failed to initialize Groq client: {e}")

    def rewrite(self, query: str) -> str:
        """
        Rewrite a user query for better retrieval.
        Falls back to original query if rewriting fails.
        """
        try:
            with Timer("query_rewrite", logger) as t:
                response = self._client.chat.completions.create(
                    model=settings.groq_model_fast,
                    messages=[
                        {
                            "role": "user",
                            "content": REWRITE_PROMPT.format(query=query),
                        }
                    ],
                    max_tokens=100,
                    temperature=0.1,  # low temp — we want consistent rewrites
                )

            rewritten = response.choices[0].message.content.strip()

            logger.info(
                "Query rewritten",
                extra={
                    "original": query,
                    "rewritten": rewritten,
                    "latency_ms": round(t.elapsed_ms, 2),
                }
            )
            return rewritten

        except Exception as e:
            # Never let rewriting break the pipeline — fall back to original
            logger.warning(
                "Query rewriting failed, using original query",
                extra={"error": str(e)}
            )
            return query


# Module-level singleton
query_rewriter = QueryRewriter()