# open-data-reranker

A lightweight reranking microservice built with FastAPI, powered by [jinaai/jina-reranker-v3](https://huggingface.co/jinaai/jina-reranker-v3) (0.6B params).

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

Higher scores indicate greater relevance.

### `GET /health`

Returns `{"status": "ok"}`.

## Docker

```bash
make build     # build image with model baked in
make run       # run container on :8000
```

The Docker build pre-downloads model weights so the container starts instantly.

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

The model weights are governed by the [Jina AI License](https://huggingface.co/jinaai/jina-reranker-v3).
