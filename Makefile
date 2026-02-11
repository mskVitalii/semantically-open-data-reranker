DATE_TAG := $(shell date +%Y-%m-%d-%H-%M)
VERSION ?= $(DATE_TAG)

# Supported models
MODELS := \
    jinaai/jina-reranker-v3

# Default model
MODEL ?= jinaai/jina-reranker-v3

# Convert model name to tag-friendly format (lowercase, replace / and special chars with -)
MODEL_TAG := $(shell echo "$(MODEL)" | tr '/' '-' | tr '[:upper:]' '[:lower:]')

# Image naming
IMAGE_BASE = mskkote/reranker
IMAGE_NAME = $(IMAGE_BASE)-$(MODEL_TAG)
PORT = 8000

.PHONY: install dev build run push test health build-all push-all list-models info

install:
	uv sync

dev:
	MODEL_NAME="$(MODEL)" uv run uvicorn app.main:app --reload --host 0.0.0.0 --port $(PORT)

build:
	@echo "Building $(IMAGE_NAME):$(VERSION) with model $(MODEL)"
	docker build \
		--build-arg MODEL_NAME="$(MODEL)" \
		-t $(IMAGE_NAME):$(VERSION) \
		-t $(IMAGE_NAME):latest \
		.

run:
	docker run --rm -p $(PORT):8000 $(IMAGE_NAME):latest

push:
	@echo "Pushing all tags for $(IMAGE_NAME)"
	@docker images --format "{{.Repository}}:{{.Tag}}" | grep "^$(IMAGE_NAME):" | while read image; do \
		echo "Pushing $$image"; \
		docker push $$image || true; \
	done

# Build all models
build-all:
	@for model in $(MODELS); do \
		echo "Building $$model..."; \
		$(MAKE) build MODEL=$$model || exit 1; \
	done
	@echo "All models built successfully!"

# Push all models
push-all:
	@for model in $(MODELS); do \
		echo "Pushing $$model..."; \
		$(MAKE) push MODEL=$$model || exit 1; \
	done
	@echo "All models pushed successfully!"

# List all supported models
list-models:
	@echo "Supported models:"
	@for model in $(MODELS); do \
		echo "  - $$model"; \
	done

# Show build information
info:
	@echo "Model: $(MODEL)"
	@echo "Model Tag: $(MODEL_TAG)"
	@echo "Image Name: $(IMAGE_NAME):$(VERSION)"
	@echo "Image Name (latest): $(IMAGE_NAME):latest"
	@echo "Date Tag: $(DATE_TAG)"

test:
	curl -X POST http://localhost:$(PORT)/rerank \
		-H "Content-Type: application/json" \
		-d '{"query": "What is machine learning?", "documents": ["Machine learning is a subset of AI", "The weather is nice today", "Deep learning uses neural networks"]}' | jq

health:
	curl http://localhost:$(PORT)/health | jq
