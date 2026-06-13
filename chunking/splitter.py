import re
from dataclasses import dataclass, field
from typing import Optional

from config.constants import (
    META_CHUNK_INDEX,
    META_TOTAL_CHUNKS,
    META_ACT_NAME,
    META_SECTION_NUMBER,
    DOC_TYPE_ACT,
    DOC_TYPE_CONSTITUTION,
)
from config.exceptions import ChunkingError
from config.settings import get_settings
from ingestion.loader import Document
from observability.logger import setup_logger

logger = setup_logger(__name__)
settings = get_settings()


@dataclass
class Chunk:
    """
    A single chunk of text ready for embedding.
    Carries its own metadata — chunk_index, section_number etc.
    """
    text: str
    metadata: dict = field(default_factory=dict)
    chunk_id: Optional[str] = None

# ── Section pattern — matches "Section 420", "Article 21", "Clause 3" ────────
SECTION_PATTERN = re.compile(
    r"((?:Section|SECTION|Article|ARTICLE|Clause|CLAUSE)\s+\d+[A-Z]?\.?)",
    re.MULTILINE,
)


def extract_section_metadata(text: str) -> dict:
    """
    Look for a section header at the start of a chunk.
    Returns act_name and section_number if found.
    """
    match = SECTION_PATTERN.search(text[:200])  # only check start of chunk
    if match:
        return {META_SECTION_NUMBER: match.group(1).strip()}
    return {}


def recursive_split(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """
    Split text using a hierarchy of separators.
    Tries paragraph breaks first, then sentences, then words.
    Only falls back to hard character cuts as last resort.
    """
    separators = ["\n\n", "\n", ". ", " ", ""]

    def split_with_separator(text: str, separators: list[str]) -> list[str]:
        if not separators:
            # Last resort — hard character cut
            return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size - chunk_overlap)]

        sep = separators[0]
        splits = text.split(sep) if sep else list(text)

        chunks = []
        current = ""

        for split in splits:
            candidate = current + sep + split if current else split

            if len(candidate) <= chunk_size:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                # If single split is bigger than chunk_size, recurse
                if len(split) > chunk_size:
                    chunks.extend(split_with_separator(split, separators[1:]))
                    current = ""
                else:
                    current = split

        if current:
            chunks.append(current)

        return chunks

    raw_chunks = split_with_separator(text, separators)

    # Apply overlap — each chunk starts with the last overlap chars of previous chunk
    if chunk_overlap == 0 or len(raw_chunks) <= 1:
        return raw_chunks

    overlapped = [raw_chunks[0]]
    for i in range(1, len(raw_chunks)):
        prev_tail = raw_chunks[i-1][-chunk_overlap:]
        overlapped.append(prev_tail + " " + raw_chunks[i])

    return overlapped


def section_aware_split(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """
    Split structured legal documents at section boundaries.
    Each chunk stays within one section.
    If a section is longer than chunk_size, falls back to recursive split.
    """
    # Split at every section header — keep the header with its content
    parts = SECTION_PATTERN.split(text)

    sections = []
    i = 0
    while i < len(parts):
        part = parts[i].strip()
        if not part:
            i += 1
            continue
        # If this part is a section header, combine with its content
        if SECTION_PATTERN.match(part) and i + 1 < len(parts):
            content = parts[i + 1].strip()
            sections.append(f"{part}\n{content}")
            i += 2
        else:
            sections.append(part)
            i += 1

    # Now split any section that's still too large
    final_chunks = []
    for section in sections:
        if len(section) <= chunk_size:
            final_chunks.append(section)
        else:
            final_chunks.extend(recursive_split(section, chunk_size, chunk_overlap))

    return [c for c in final_chunks if c.strip()]


def split_document(document: Document) -> list[Chunk]:
    """
    Main entry point for chunking.
    Chooses strategy based on doc_type then returns list of Chunk objects.
    """
    if not document.text:
        raise ChunkingError("Document has no text to chunk.")

    chunk_size = settings.chunk_size
    chunk_overlap = settings.chunk_overlap
    doc_type = document.metadata.get("doc_type", "generic")

    logger.info(
        "Chunking document",
        extra={
            "file": document.metadata.get("source"),
            "doc_type": doc_type,
            "chunk_size": chunk_size,
        },
    )

    # Choose strategy
    if doc_type in (DOC_TYPE_ACT, DOC_TYPE_CONSTITUTION):
        raw_chunks = section_aware_split(document.text, chunk_size, chunk_overlap)
    else:
        raw_chunks = recursive_split(document.text, chunk_size, chunk_overlap)

    if not raw_chunks:
        raise ChunkingError(f"No chunks produced for {document.metadata.get('source')}")

    # Build Chunk objects with metadata
    chunks = []
    total = len(raw_chunks)

    for i, text in enumerate(raw_chunks):
        metadata = {
            **document.metadata,           # inherit all document metadata
            META_CHUNK_INDEX: i,
            META_TOTAL_CHUNKS: total,
            **extract_section_metadata(text),  # add section number if found
        }
        chunk = Chunk(
            text=text.strip(),
            metadata=metadata,
            chunk_id=f"{document.metadata.get('file_hash', 'doc')}_{i}",
        )
        chunks.append(chunk)

    logger.info(
        "Chunking complete",
        extra={
            "file": document.metadata.get("source"),
            "total_chunks": total,
        },
    )

    return chunks   