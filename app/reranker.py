import gc
import logging
import math
import os
import threading

import torch

logger = logging.getLogger(__name__)
from transformers import AutoModel

MODEL_NAME = os.environ.get("MODEL_NAME", "jinaai/jina-reranker-v3")

_model = None
_model_lock = threading.Lock()
_device = "cpu"


def _get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _sync_and_empty_cache() -> None:
    """Synchronize device and release cached memory."""
    if _device == "mps":
        torch.mps.synchronize()
        torch.mps.empty_cache()
    elif _device == "cuda":
        torch.cuda.empty_cache()
    gc.collect()


def load_model() -> None:
    global _model, _device
    _device = _get_device()
    logger.info("Using device: %s", _device)
    _model = AutoModel.from_pretrained(
        MODEL_NAME,
        dtype="auto",
        trust_remote_code=True,
    ).to(_device)
    _model.requires_grad_(False)


MAX_BATCH_SIZE = int(os.environ.get("RERANK_BATCH_SIZE", "10"))


def _rerank_batch(query: str, documents: list[str], top_n: int | None, batch_size: int) -> list[dict]:
    """Rerank documents in batches to avoid OOM, then merge and sort results."""
    all_results: list[dict] = []

    for start in range(0, len(documents), batch_size):
        batch_docs = documents[start : start + batch_size]
        with torch.no_grad():
            batch_results = _model.rerank(query, batch_docs, top_n=len(batch_docs))
        for r in batch_results:
            all_results.append({
                "index": r["index"] + start,
                "document": r["document"],
                "relevance_score": r["relevance_score"],
            })
        _sync_and_empty_cache()

    all_results.sort(key=lambda r: r["relevance_score"], reverse=True)
    if top_n is not None:
        all_results = all_results[:top_n]
    return all_results


def rerank(
    query: str,
    documents: list[str],
    top_n: int | None = None,
) -> list[dict]:
    if _model is None:
        raise RuntimeError("Model not loaded. Call load_model() first.")

    with _model_lock:
        batch_size = MAX_BATCH_SIZE
        while batch_size >= 1:
            try:
                results = _rerank_batch(query, documents, top_n, batch_size)
                break
            except RuntimeError as e:
                if "out of memory" in str(e).lower() and batch_size > 1:
                    _sync_and_empty_cache()
                    batch_size = max(1, batch_size // 2)
                    logger.warning("OOM with batch_size=%d, retrying with %d", batch_size * 2, batch_size)
                else:
                    raise
    return [
        {
            "index": r["index"],
            "document": r["document"],
            "relevance_score": 0.0 if math.isnan(r["relevance_score"]) else round(r["relevance_score"], 6),
        }
        for r in results
    ]
