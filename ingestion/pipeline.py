import re
from config.exceptions import DuplicateDocumentError
from config.constants import META_FILE_HASH
from config.settings import get_settings
from ingestion.loader import load_document
from chunking.splitter import split_document
from embeddings.embedder import embedder
from vectorstore.store import vector_store
from retrieval.bm25_index import bm25_index
from observability.logger import setup_logger, Timer
from observability.cache import query_cache

logger = setup_logger(__name__)
settings = get_settings()


def clean_chunk_text(text: str) -> str:
    """Fix remaining PDF artifacts in chunk text before embedding."""
    text = re.sub(r"([A-Z])\n([a-z])", r"\1\2", text)
    text = re.sub(r"([a-z])\n([a-z])", r"\1 \2", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def ingest_document(file_path: str, doc_type: str) -> dict:
    """
    Full ingestion pipeline:
    load → chunk → embed → store → rebuild BM25 index
    """
    with Timer("full_ingestion", logger) as t:

        # Step 1 — Load
        document = load_document(file_path, doc_type)

        # Step 2 — Dedup check
        file_hash = document.metadata[META_FILE_HASH]
        if vector_store.document_exists(file_hash):
            raise DuplicateDocumentError(
                f"Document already ingested: {document.metadata['source']}"
            )

        # Step 3 — Chunk
        chunks = split_document(document)

        # Step 4 — Embed
        texts = [c.text for c in chunks]
        vectors = embedder.embed_texts(texts)

        # Step 5 — Store in Qdrant
        chunk_ids = [c.chunk_id for c in chunks]
        metadata_list = []
        for chunk, text in zip(chunks, texts):
            clean_text = clean_chunk_text(text)
            meta = {**chunk.metadata, "text": clean_text}
            metadata_list.append(meta)

        vector_store.upsert(chunk_ids, vectors, metadata_list)

        # Step 6 — Rebuild BM25 index
        bm25_index.rebuild_from_qdrant(
            vector_store._client,
            settings.qdrant_collection_name
        )

    logger.info(
        "Ingestion pipeline complete",
        extra={
            "file": document.metadata["source"],
            "chunks": len(chunks),
            "latency_ms": round(t.elapsed_ms, 2),
        }
    )
    # Invalidate query cache — new document means old cached answers may be incomplete
    query_cache.invalidate()

    return {
        "source": document.metadata["source"],
        "doc_type": doc_type,
        "chunks_ingested": len(chunks),
        "file_hash": file_hash,
    }