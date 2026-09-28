import json
import os
import shutil
import tempfile
import threading
import uuid
from pathlib import Path
from datetime import datetime, timezone
from fastapi import APIRouter

from api.schemas import ReviewItem, ReviewDecision, Source, QueryResponse
from config.settings import get_settings
from observability.logger import setup_logger

logger = setup_logger(__name__)
settings = get_settings()
router = APIRouter(prefix="/review", tags=["Human Review"])

QUEUE_FILE = Path("./data/processed/review_queue.json")
APPROVED_FILE = Path("./data/processed/approved_answers.json")

# Guards read-modify-write cycles on the queue/approved JSON files. Without
# this, two concurrent requests can both read the same list, both append/
# modify independently, and whichever write happens last silently discards
# the other's change (lost update).
_queue_lock = threading.Lock()
_approved_lock = threading.Lock()


def _atomic_write_json(path: Path, data):
    """
    Write via temp file + atomic rename so a crash or kill mid-write can't
    leave a truncated, unparseable JSON file behind.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=str(path.parent))
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2)
        shutil.move(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def _load_queue() -> list[dict]:
    if not QUEUE_FILE.exists():
        return []
    return json.loads(QUEUE_FILE.read_text())


def _save_queue(queue: list[dict]):
    _atomic_write_json(QUEUE_FILE, queue)


def _load_approved() -> list[dict]:
    if not APPROVED_FILE.exists():
        return []
    return json.loads(APPROVED_FILE.read_text())


def _save_approved(approved: list[dict]):
    _atomic_write_json(APPROVED_FILE, approved)


def add_to_review_queue(
    query: str,
    rewritten_query: str,
    chunks: list[dict],
    draft_answer: str,
    sources: list[dict],
) -> str:
    """
    Add a low-confidence response to the review queue.
    Returns the review_id.
    Called from the query pipeline when score is below review_threshold.
    """
    review_id = str(uuid.uuid4())[:8]
    item = {
        "review_id": review_id,
        "query": query,
        "rewritten_query": rewritten_query,
        "retrieved_chunks": [
            {
                "text": c["text"],
                "section": c["metadata"].get("section_number", ""),
                "source": c["metadata"].get("source", ""),
                "rerank_score": c.get("rerank_score", 0),
            }
            for c in chunks
        ],
        "draft_answer": draft_answer,
        "sources": sources,
        "status": "pending",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    with _queue_lock:
        queue = _load_queue()
        queue.append(item)
        _save_queue(queue)

    logger.info(
        "Added to review queue",
        extra={"review_id": review_id, "query": query[:80]}
    )
    return review_id


@router.get("/pending", response_model=list[ReviewItem])
async def get_pending():
    """Get all pending review items."""
    queue = _load_queue()
    pending = [q for q in queue if q["status"] == "pending"]
    return pending


@router.post("/decide")
async def decide(decision: ReviewDecision):
    """
    Approve or correct a review item.
    Approved answers are saved to the golden dataset automatically.
    """
    with _queue_lock:
        queue = _load_queue()
        item = next((q for q in queue if q["review_id"] == decision.review_id), None)

        if not item:
            return {"error": f"Review item {decision.review_id} not found"}

        # Update status
        item["status"] = "approved" if decision.approved else "rejected"
        item["reviewed_at"] = datetime.now(timezone.utc).isoformat()
        item["reviewer_note"] = decision.reviewer_note

        final_answer = decision.corrected_answer or item["draft_answer"]
        item["final_answer"] = final_answer

        _save_queue(queue)

    # If approved — save to golden dataset for evaluation
    if decision.approved:
        with _approved_lock:
            approved = _load_approved()
            approved.append({
                "question": item["query"],
                "rewritten_query": item["rewritten_query"],
                "answer": final_answer,
                "contexts": [c["text"] for c in item["retrieved_chunks"]],
                "sources": item["sources"],
                "approved_at": item["reviewed_at"],
            })
            _save_approved(approved)

        logger.info(
            "Answer approved and added to golden dataset",
            extra={"review_id": decision.review_id}
        )

    return {
        "review_id": decision.review_id,
        "status": item["status"],
        "final_answer": final_answer,
    }



@router.get("/approved")
async def get_approved():
    """Get all approved answers — this is your golden dataset."""
    return _load_approved()