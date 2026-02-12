import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.reranker import load_model, rerank
from app.schemas import RerankRequest, RerankResponse, RerankResult

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield


app = FastAPI(title="Reranker Service", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/rerank", response_model=RerankResponse)
def rerank_endpoint(request: RerankRequest):
    logger.info("Rerank request: query=%r, documents=%d", request.query, len(request.documents))
    start = time.perf_counter()
    raw_results = rerank(request.query, request.documents, top_n=request.top_n)
    elapsed = time.perf_counter() - start
    logger.info("Rerank done in %.3fs", elapsed)
    results = [
        RerankResult(
            index=r["index"],
            document=r["document"],
            relevance_score=r["relevance_score"],
        )
        for r in raw_results
    ]
    return RerankResponse(results=results)
