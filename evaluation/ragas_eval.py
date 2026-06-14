import json
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from groq import Groq
from config.settings import get_settings
from observability.logger import setup_logger
from observability.mlflow_tracker import log_eval_metrics

logger = setup_logger(__name__)
settings = get_settings()
client = Groq(api_key=settings.groq_api_key)


def load_golden_dataset(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def run_pipeline_on_question(question: str) -> tuple[str, list[str]]:
    from embeddings.embedder import embedder
    from vectorstore.store import vector_store
    from retrieval.retriever import retriever
    from retrieval.reranker import reranker
    from generation.query_rewriter import query_rewriter
    from generation.generator import generator

    embedder.load()
    vector_store.connect()
    reranker.load()

    rewritten = query_rewriter.rewrite(question)
    chunks = retriever.retrieve(rewritten)
    reranked = reranker.rerank(rewritten, chunks)
    result = generator.generate(rewritten, reranked)
    contexts = [c["text"] for c in reranked]
    return result["answer"], contexts


def llm_judge(prompt: str) -> str:
    """Call Groq as an LLM judge."""
    response = client.chat.completions.create(
        model=settings.groq_model_fast,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=10,
        temperature=0.0,
    )
    return response.choices[0].message.content.strip().lower()


def score_faithfulness(answer: str, contexts: list[str]) -> float:
    """
    Faithfulness — what fraction of answer sentences are 
    supported by the retrieved contexts.
    """
    sentences = [s.strip() for s in answer.split(".") if len(s.strip()) > 20]
    if not sentences:
        return 0.0

    context_text = "\n".join(contexts)
    supported = 0

    for sentence in sentences:
        prompt = f"""Given the following context:
{context_text}

Is this statement supported by the context? Answer only 'yes' or 'no':
"{sentence}"
"""
        answer_txt = llm_judge(prompt)
        if "yes" in answer_txt:
            supported += 1

    return supported / len(sentences)


def score_answer_relevancy(question: str, answer: str) -> float:
    """
    Answer relevancy — cosine similarity between question 
    and answer embeddings.
    """
    from embeddings.embedder import embedder
    # Model already loaded by pipeline — just use it
    q_vec = embedder.embed_query(question)
    a_vec = embedder.embed_query(answer)
    return float(np.dot(q_vec, a_vec))

def score_context_precision(question: str, contexts: list[str]) -> float:
    """
    Context precision — what fraction of retrieved chunks 
    are actually relevant to the question.
    """
    if not contexts:
        return 0.0

    relevant = 0
    for ctx in contexts:
        prompt = f"""Question: {question}

Context: {ctx}

Is this context relevant to answering the question? Answer only 'yes' or 'no':"""
        result = llm_judge(prompt)
        if "yes" in result:
            relevant += 1

    return relevant / len(contexts)


def score_context_recall(ground_truth: str, contexts: list[str]) -> float:
    """
    Context recall — does the retrieved context contain 
    the information needed for the ground truth answer.
    """
    context_text = "\n".join(contexts)
    prompt = f"""Ground truth answer: {ground_truth}

Retrieved context:
{context_text}

Does the retrieved context contain enough information to arrive at the ground truth answer? Answer only 'yes' or 'no':"""

    result = llm_judge(prompt)
    return 1.0 if "yes" in result else 0.0

def run_evaluation(golden_path: str = None):
    golden_path = golden_path or settings.golden_dataset_path
    logger.info("Starting evaluation", extra={"dataset": golden_path})

    golden = load_golden_dataset(golden_path)
    logger.info(f"Loaded {len(golden)} golden examples")

    all_faithfulness = []
    all_relevancy = []
    all_precision = []
    all_recall = []

    # Pre-load all components once before the evaluation loop
    from embeddings.embedder import embedder
    from vectorstore.store import vector_store
    from retrieval.reranker import reranker
    embedder.load()
    vector_store.connect()
    reranker.load()
    logger.info("Components pre-loaded")

    for item in golden:
        question = item["question"]
        ground_truth = item["ground_truth"]

        logger.info(f"Evaluating: {question}")

        try:
            answer, contexts = run_pipeline_on_question(question)

            f = score_faithfulness(answer, contexts)
            r = score_answer_relevancy(question, answer)
            p = score_context_precision(question, contexts)
            rc = score_context_recall(ground_truth, contexts)

            all_faithfulness.append(f)
            all_relevancy.append(r)
            all_precision.append(p)
            all_recall.append(rc)

            print(f"\nQ: {question}")
            print(f"  Faithfulness:      {f:.4f}")
            print(f"  Answer Relevancy:  {r:.4f}")
            print(f"  Context Precision: {p:.4f}")
            print(f"  Context Recall:    {rc:.4f}")

        except Exception as e:
            logger.warning(f"Eval failed for '{question}': {e}")
            continue

    if not all_faithfulness:
        logger.error("No questions evaluated successfully")
        return

    scores = {
        "faithfulness": float(np.mean(all_faithfulness)),
        "answer_relevancy": float(np.mean(all_relevancy)),
        "context_precision": float(np.mean(all_precision)),
        "context_recall": float(np.mean(all_recall)),
    }

    print("\n" + "="*50)
    print("Evaluation Results (averaged)")
    print("="*50)
    for metric, score in scores.items():
        print(f"{metric:25s}: {score:.4f}")
    print("="*50)

    log_eval_metrics(scores)
    return scores


if __name__ == "__main__":
    run_evaluation()