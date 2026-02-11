import os

from transformers import AutoModel

MODEL_NAME = os.environ.get("MODEL_NAME", "jinaai/jina-reranker-v3")

_model = None


def load_model() -> None:
    global _model
    _model = AutoModel.from_pretrained(
        MODEL_NAME,
        dtype="auto",
        trust_remote_code=True,
    )
    _model.eval()


def rerank(
    query: str,
    documents: list[str],
    top_n: int | None = None,
) -> list[dict]:
    if _model is None:
        raise RuntimeError("Model not loaded. Call load_model() first.")

    results = _model.rerank(query, documents, top_n=top_n)
    return [
        {
            "index": r["index"],
            "document": r["document"],
            "relevance_score": round(r["relevance_score"], 6),
        }
        for r in results
    ]
