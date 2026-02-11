FROM python:3.12-slim

# Build argument for model name
ARG MODEL_NAME=nvidia/llama-nemotron-rerank-vl-1b-v2

# Environment variables
ENV MODEL_NAME=${MODEL_NAME}
ENV HF_HOME=/root/.cache/huggingface

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --no-dev --no-install-project

COPY app/ app/

# Pre-download the model during build (imports app.reranker to apply the flash_attn patch)
RUN uv run python -c "from app.reranker import load_model; load_model()"

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
