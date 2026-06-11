class LexRAGError(Exception):
    """Base exception for all LexRAG errors."""
    pass


# Ingestion
class IngestionError(LexRAGError):
    pass

class UnsupportedFileTypeError(IngestionError):
    pass

class DuplicateDocumentError(IngestionError):
    pass


# Chunking
class ChunkingError(LexRAGError):
    pass


# Embedding
class EmbeddingError(LexRAGError):
    pass

class EmbeddingModelNotLoadedError(EmbeddingError):
    pass


# Vector Store
class VectorStoreError(LexRAGError):
    pass

class CollectionNotFoundError(VectorStoreError):
    pass


# Retrieval
class RetrievalError(LexRAGError):
    pass

# BelowConfidenceThreshold
class BelowConfidenceThresholdError(RetrievalError):
    """
    Raised when no retrieved chunk meets the minimum similarity threshold.
    Signals the API to return 'insufficient information' instead of hallucinating.
    """
    pass


# Generation
class GenerationError(LexRAGError):
    pass

class LLMConnectionError(GenerationError):
    pass

class ContextWindowExceededError(GenerationError):
    pass


# Evaluation
class EvaluationError(LexRAGError):
    pass