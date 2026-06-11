from pydantic_settings import BaseSettings
from pydantic import Field
from functools import lru_cache
from pathlib import Path


class Settings(BaseSettings):
    # LLM
    groq_api_key: str = Field(..., env="GROQ_API_KEY")
    groq_model_fast: str = Field("llama3-8b-8192", env="GROQ_MODEL_FAST")
    groq_model_quality: str = Field("llama3-70b-8192", env="GROQ_MODEL_QUALITY")

    # Embeddings
    embedding_model: str = Field("BAAI/bge-base-en-v1.5", env="EMBEDDING_MODEL")
    embedding_device: str = Field("cpu", env="EMBEDDING_DEVICE")
    embedding_batch_size: int = Field(32, env="EMBEDDING_BATCH_SIZE")

    # Vector Store
    qdrant_host: str = Field("localhost", env="QDRANT_HOST")
    qdrant_port: int = Field(6333, env="QDRANT_PORT")
    qdrant_collection_name: str = Field("lexrag_legal", env="QDRANT_COLLECTION_NAME")
    qdrant_vector_size: int = Field(768, env="QDRANT_VECTOR_SIZE")

    # Retrieval
    dense_top_k: int = Field(20, env="DENSE_TOP_K")
    sparse_top_k: int = Field(20, env="SPARSE_TOP_K")
    final_top_k: int = Field(5, env="FINAL_TOP_K")
    rerank_model: str = Field("cross-encoder/ms-marco-MiniLM-L-6-v2", env="RERANK_MODEL")
    min_similarity_threshold: float = Field(0.30, env="MIN_SIMILARITY_THRESHOLD")

    # Chunking
    chunk_size: int = Field(512, env="CHUNK_SIZE")
    chunk_overlap: int = Field(50, env="CHUNK_OVERLAP")

    # MLflow
    mlflow_tracking_uri: str = Field("./data/mlflow", env="MLFLOW_TRACKING_URI")
    mlflow_experiment_name: str = Field("lexrag_experiments", env="MLFLOW_EXPERIMENT_NAME")

    # API
    api_host: str = Field("0.0.0.0", env="API_HOST")
    api_port: int = Field(8000, env="API_PORT")
    log_level: str = Field("INFO", env="LOG_LEVEL")

    # Evaluation
    golden_dataset_path: str = Field("./data/processed/golden_qa.json", env="GOLDEN_DATASET_PATH")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

    def ensure_dirs(self):
        dirs = [
            "./data/raw",
            "./data/processed",
            "./data/logs",
            "./data/mlflow",
        ]
        for d in dirs:
            Path(d).mkdir(parents=True, exist_ok=True)


@lru_cache()
def get_settings() -> Settings:
    return Settings()