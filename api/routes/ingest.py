from fastapi import APIRouter, UploadFile, File, Form
from pathlib import Path
import shutil

from api.schemas import IngestResponse
from config.constants import SUPPORTED_EXTENSIONS, DOC_TYPE_GENERIC
from config.settings import get_settings
from config.exceptions import UnsupportedFileTypeError
from ingestion.pipeline import ingest_document
from observability.logger import setup_logger
from observability.metrics import INGEST_COUNTER, update_chunks_gauge

logger = setup_logger(__name__)
settings = get_settings()
router = APIRouter(prefix="/ingest", tags=["Ingestion"])


@router.post("", response_model=IngestResponse)
async def ingest(
    file: UploadFile = File(...),
    doc_type: str = Form(default=DOC_TYPE_GENERIC),
):
    """
    Upload and ingest a legal document.
    Supported formats: PDF, TXT, DOCX.
    """
    ext = Path(file.filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFileTypeError(
            f"Unsupported file type '{ext}'. Supported: {SUPPORTED_EXTENSIONS}"
        )

    temp_path = Path(f"./data/raw/{file.filename}")
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    logger.info("File uploaded", extra={"file": file.filename, "doc_type": doc_type})

    result = ingest_document(str(temp_path), doc_type)

    # Track metrics
    INGEST_COUNTER.labels(doc_type=doc_type).inc()

    # Update chunks gauge
    from vectorstore.store import vector_store
    try:
        collection_info = vector_store._client.get_collection(
            settings.qdrant_collection_name
        )
        update_chunks_gauge(collection_info.points_count)
    except Exception:
        pass

    return IngestResponse(**result)