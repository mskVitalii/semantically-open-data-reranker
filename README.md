# open-data-reranker

A lightweight reranking microservice built with FastAPI, powered by [nvidia/llama-nemotron-rerank-vl-1b-v2](https://huggingface.co/nvidia/llama-nemotron-rerank-vl-1b-v2) (1.7B params).

Takes a query and a list of documents, returns them ranked by relevance.

## Quickstart

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
make install   # install dependencies
make dev       # start dev server on :8000
```

Test it:

```bash
curl -X POST http://localhost:8000/rerank \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is machine learning?",
    "documents": [
      "Machine learning is a subset of AI",
      "The weather is nice today",
      "Deep learning uses neural networks"
    ]
  }' | jq
```

## API

### `POST /rerank`

Rerank documents by relevance to a query.

**Request:**

| Field       | Type       | Required | Description                          |
|-------------|------------|----------|--------------------------------------|
| `query`     | `string`   | yes      | The search query                     |
| `documents` | `string[]` | yes      | Documents to rank                    |
| `top_n`     | `int`      | no       | Return only the top N results        |

**Response:**

```json
{
  "results": [
    {
      "index": 0,
      "document": "Machine learning is a subset of AI",
      "relevance_score": 0.987654
    },
    {
      "index": 2,
      "document": "Deep learning uses neural networks",
      "relevance_score": 0.876543
    },
    {
      "index": 1,
      "document": "The weather is nice today",
      "relevance_score": 0.012345
    }
  ]
}
```

Scores are sigmoid-normalized logits in the `[0, 1]` range.

### `GET /health`

Returns `{"status": "ok"}`.

## Docker

```bash
make build     # build image with model baked in
make run       # run container on :8000
```

The Docker build pre-downloads model weights so the container starts instantly.

## GPU support

On CUDA systems the service automatically uses `bfloat16` precision and `flash_attention_2`. On CPU it falls back to `float32` with eager attention.

For GPU Docker builds, swap the base image to a CUDA-enabled one and add `flash-attn`:

```dockerfile
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04
# ...
RUN pip install "flash-attn>=2.6.3,<2.8" --no-build-isolation
```

## Project structure

```
app/
  main.py       # FastAPI app, lifespan, endpoints
  reranker.py   # Model loading and inference
  schemas.py    # Pydantic request/response models
Dockerfile      # Production container (model pre-downloaded)
Makefile        # Dev/build/deploy commands
```

## License

The model weights are governed by the [NVIDIA Open Model License](https://developer.nvidia.com/open-model-license). Built with Llama.
