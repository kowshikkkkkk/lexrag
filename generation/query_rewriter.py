from groq import Groq

from config.settings import get_settings
from config.exceptions import GenerationError, LLMConnectionError
from observability.logger import setup_logger, Timer

logger = setup_logger(__name__)
settings = get_settings()

REWRITE_PROMPT = """You are a legal query optimizer for Indian law.
Rewrite the user's question into a clear query for searching legal documents.

STRICT RULES:
- Expand abbreviations only (IPC → Indian Penal Code, CrPC → Code of Criminal Procedure)
- Add general legal context if missing
- NEVER add section numbers unless the user explicitly stated one
- If user said "420" treat it as a section number they mentioned — keep it
- If user did NOT mention a section number — do NOT add one
- One sentence only
- Return ONLY the rewritten query, nothing else

Examples:
"what is cheating under IPC" → "What is the definition and punishment of cheating under the Indian Penal Code?"
"what does 420 mean in law" → "What is the meaning of Section 420 of the Indian Penal Code?"
"punishment for stealing" → "What is the punishment for theft under the Indian Penal Code?"

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