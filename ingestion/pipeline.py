from config.exceptions import DuplicateDocumentError
from config.constants import META_FILE_HASH
from ingestion.loader import load_document
from chunking.splitter import split_document
from embeddings.embedder import embedder
from vectorstore.store import vector_store
from observability.logger import setup_logger, Timer

logger = setup_logger(__name__)


def ingest_document(file_path: str, doc_type: str) -> dict:
    """
    Full ingestion pipeline:
    load → chunk → embed → store

    Returns summary of what was ingested.
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

        # Step 5 — Store
        # Add chunk text to metadata so we can retrieve it from Qdrant
        chunk_ids = [c.chunk_id for c in chunks]
        metadata_list = []
        for chunk, text in zip(chunks, texts):
            meta = {**chunk.metadata, "text": text}
            metadata_list.append(meta)

        vector_store.upsert(chunk_ids, vectors, metadata_list)

    logger.info(
        "Ingestion pipeline complete",
        extra={
            "file": document.metadata["source"],
            "chunks": len(chunks),
            "latency_ms": round(t.elapsed_ms, 2),
        }
    )

    return {
        "source": document.metadata["source"],
        "doc_type": doc_type,
        "chunks_ingested": len(chunks),
        "file_hash": file_hash,
    }