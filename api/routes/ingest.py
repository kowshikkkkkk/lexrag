from fastapi import APIRouter, UploadFile, File, Form
from pathlib import Path
import shutil

from api.schemas import IngestResponse
from config.constants import SUPPORTED_EXTENSIONS, DOC_TYPE_GENERIC
from config.exceptions import UnsupportedFileTypeError
from ingestion.pipeline import ingest_document
from observability.logger import setup_logger

logger = setup_logger(__name__)
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
    # Validate file type
    ext = Path(file.filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFileTypeError(
            f"Unsupported file type '{ext}'. Supported: {SUPPORTED_EXTENSIONS}"
        )

    # Save uploaded file temporarily
    temp_path = Path(f"./data/raw/{file.filename}")
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    logger.info("File uploaded", extra={"file": file.filename, "doc_type": doc_type})

    # Run ingestion pipeline
    result = ingest_document(str(temp_path), doc_type)

    return IngestResponse(**result)