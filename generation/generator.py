from groq import Groq
from typing import Iterator

from config.settings import get_settings
from config.constants import (
    SYSTEM_PROMPT,
    INSUFFICIENT_INFO_RESPONSE,
    MAX_CONTEXT_TOKENS,
    TOKENS_PER_WORD,
)
from config.exceptions import GenerationError, LLMConnectionError, ContextWindowExceededError
from observability.logger import setup_logger, Timer

logger = setup_logger(__name__)
settings = get_settings()


def _build_context(chunks: list[dict]) -> tuple[str, list[dict]]:
    """
    Build context string from chunks respecting token budget.
    Returns the context string and the list of chunks that fit.
    """
    context_parts = []
    used_chunks = []
    total_tokens = 0

    for chunk in chunks:
        text = chunk["text"]
        # Rough token estimate
        chunk_tokens = len(text.split()) * TOKENS_PER_WORD

        if total_tokens + chunk_tokens > MAX_CONTEXT_TOKENS:
            logger.warning(
                "Token budget reached, truncating context",
                extra={"chunks_used": len(used_chunks), "chunks_skipped": len(chunks) - len(used_chunks)}
            )
            break

        source = chunk["metadata"].get("source", "Unknown")
        section = chunk["metadata"].get("section_number", "")
        citation = f"[Source: {source}" + (f", {section}]" if section else "]")

        context_parts.append(f"{citation}\n{text}")
        used_chunks.append(chunk)
        total_tokens += chunk_tokens

    return "\n\n---\n\n".join(context_parts), used_chunks


def _build_prompt(query: str, context: str) -> str:
    return f"""Context documents:
{context}

Question: {query}

Answer:"""


class Generator:
    """
    Handles LLM generation with guardrails.
    Supports both standard and streaming responses.
    """

    def __init__(self):
        try:
            self._client = Groq(api_key=settings.groq_api_key)
        except Exception as e:
            raise LLMConnectionError(f"Failed to initialize Groq client: {e}")

    def generate(self, query: str, chunks: list[dict]) -> dict:
        """
        Standard (non-streaming) generation.
        Returns answer string plus source metadata.
        """
        if not chunks:
            return {
                "answer": INSUFFICIENT_INFO_RESPONSE,
                "sources": [],
                "model": settings.groq_model_quality,
            }

        context, used_chunks = _build_context(chunks)

        if not context:
            raise ContextWindowExceededError("No chunks fit within token budget.")

        prompt = _build_prompt(query, context)

        try:
            with Timer("llm_generation", logger) as t:
                response = self._client.chat.completions.create(
                    model=settings.groq_model_quality,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=1024,
                    temperature=0.1,
                )

            answer = response.choices[0].message.content.strip()

            # Extract sources from used chunks
            sources = []
            for chunk in used_chunks:
                source = {
                    "document": chunk["metadata"].get("source", "Unknown"),
                    "section": chunk["metadata"].get("section_number", ""),
                    "doc_type": chunk["metadata"].get("doc_type", ""),
                    "rerank_score": chunk.get("rerank_score", 0),
                }
                if source not in sources:
                    sources.append(source)

            logger.info(
                "Generation complete",
                extra={
                    "query": query[:80],
                    "answer_len": len(answer),
                    "sources": len(sources),
                    "latency_ms": round(t.elapsed_ms, 2),
                }
            )

            return {
                "answer": answer,
                "sources": sources,
                "model": settings.groq_model_quality,
            }

        except Exception as e:
            raise GenerationError(f"LLM generation failed: {e}")

    def stream(self, query: str, chunks: list[dict]) -> Iterator[str]:
        """
        Streaming generation — yields tokens as they arrive from Groq.
        Used by the SSE endpoint in the API layer.
        """
        if not chunks:
            yield INSUFFICIENT_INFO_RESPONSE
            return

        context, _ = _build_context(chunks)

        if not context:
            raise ContextWindowExceededError("No chunks fit within token budget.")

        prompt = _build_prompt(query, context)

        try:
            stream = self._client.chat.completions.create(
                model=settings.groq_model_quality,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=1024,
                temperature=0.1,
                stream=True,  # key difference
            )

            logger.info("Streaming generation started", extra={"query": query[:80]})

            for chunk in stream:
                token = chunk.choices[0].delta.content
                if token:
                    yield token

        except Exception as e:
            raise GenerationError(f"Streaming generation failed: {e}")


# Module-level singleton
generator = Generator()