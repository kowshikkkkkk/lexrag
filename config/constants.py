"""
LexRAG constants.
No magic numbers scattered across the codebase — everything lives here.
"""

# ── Supported file types ─────────────────────────────────────────────────────
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".docx"}

# ── Document metadata keys ───────────────────────────────────────────────────
META_SOURCE = "source"
META_DOC_TYPE = "doc_type"
META_FILE_HASH = "file_hash"
META_CHUNK_INDEX = "chunk_index"
META_TOTAL_CHUNKS = "total_chunks"
META_INGESTED_AT = "ingested_at"
META_SECTION = "section"          # e.g. "Part II, Chapter 3"
META_ACT_NAME = "act_name"        # e.g. "Indian Penal Code"
META_SECTION_NUMBER = "section_number"  # e.g. "Section 420"

# ── Legal document types ─────────────────────────────────────────────────────
DOC_TYPE_ACT = "act"
DOC_TYPE_JUDGMENT = "judgment"
DOC_TYPE_CONSTITUTION = "constitution"
DOC_TYPE_REGULATION = "regulation"
DOC_TYPE_GENERIC = "generic"

# ── Retrieval ────────────────────────────────────────────────────────────────
RRF_K = 60              # RRF fusion constant (standard value)
MAX_CONTEXT_TOKENS = 3000   # max tokens to send to LLM as context
TOKENS_PER_WORD = 1.3       # rough estimate for token budget

# ── Generation ───────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are LexRAG, a legal research assistant specializing in Indian law.

Your answers must:
1. Be grounded ONLY in the provided context documents.
2. Cite the exact source (Act name, Section number, or case name) for every claim.
3. If the context does not contain enough information to answer, respond exactly with:
   "I do not have sufficient information in the provided documents to answer this question."
4. Never speculate or use knowledge outside the provided context.
5. Use precise legal language.

Format your response as:
- A direct answer to the question
- Supporting citations in [Source: <document name>, <section/page>] format
"""

INSUFFICIENT_INFO_RESPONSE = (
    "I do not have sufficient information in the provided documents "
    "to answer this question."
)

# ── Observability ────────────────────────────────────────────────────────────
LOG_QUERY_EVENT = "query_processed"
LOG_INGEST_EVENT = "document_ingested"
LOG_EVAL_EVENT = "evaluation_run"

# ── API ──────────────────────────────────────────────────────────────────────
API_VERSION = "v1"
API_PREFIX = f"/api/{API_VERSION}"