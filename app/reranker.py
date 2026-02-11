import os

import torch
from transformers import AutoModelForSequenceClassification, AutoProcessor
from transformers.models.llama.modeling_llama import LlamaModel

MODEL_NAME = os.environ.get("MODEL_NAME", "nvidia/llama-nemotron-rerank-vl-1b-v2")

_model = None
_processor = None

# The NVIDIA model's custom code hardcodes flash_attention_2 in its __init__
# (modeling_llama_nemotron_vl.py:255), which crashes on non-CUDA systems.
# Patch LlamaModel.__init__ (parent of LlamaBidirectionalModel) to fall back
# to eager attention when flash_attention_2 is requested but CUDA is absent.
_orig_llama_init = LlamaModel.__init__


def _llama_init_fix_attn(self, config, *args, **kwargs):
    if (
        getattr(config, "_attn_implementation", None) == "flash_attention_2"
        and not torch.cuda.is_available()
    ):
        config._attn_implementation = "eager"
    _orig_llama_init(self, config, *args, **kwargs)


LlamaModel.__init__ = _llama_init_fix_attn


def load_model() -> None:
    global _model, _processor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    attn_impl = "flash_attention_2" if torch.cuda.is_available() else "eager"

    _model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        trust_remote_code=True,
        attn_implementation=attn_impl,
    )
    _model.to(device)
    _model.eval()

    _processor = AutoProcessor.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        max_input_tiles=6,
        use_thumbnail=True,
        rerank_max_length=8192,
    )


def rerank(
    query: str,
    documents: list[str],
    top_n: int | None = None,
) -> list[dict]:
    if _model is None or _processor is None:
        raise RuntimeError("Model not loaded. Call load_model() first.")

    device = next(_model.parameters()).device

    examples = [
        {"question": query, "doc_text": doc, "doc_image": ""}
        for doc in documents
    ]

    batch_dict = _processor.process_queries_documents_crossencoder(examples)
    batch_dict = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in batch_dict.items()
    }

    with torch.no_grad():
        outputs = _model(**batch_dict, return_dict=True)
        logits = outputs.logits.squeeze(-1).float().cpu()

    scores = torch.sigmoid(logits)
    sorted_indices = torch.argsort(scores, descending=True)

    if top_n is not None:
        sorted_indices = sorted_indices[:top_n]

    return [
        {
            "index": (i := idx.item()),
            "document": documents[i],
            "relevance_score": round(scores[i].item(), 6),
        }
        for idx in sorted_indices
    ]
