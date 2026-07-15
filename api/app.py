from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field

from database.settings import settings
from retrieval.db_search import DbNewsSearchService, SearchMode


app = FastAPI(title="Incident-to-News Semantic Search")
_SEARCH_SERVICE = DbNewsSearchService()
_METRIC_FILES = {
    "linked_validation": Path("evaluation/validation_results.json"),
    "blind_validation": Path("evaluation/qrels_validation_results.json"),
    "embedding_analysis": Path("evaluation/embedding_analysis.json"),
    "benchmark_search": Path("evaluation/benchmark_results.json"),
    "benchmark_real": Path("evaluation/benchmark_real_results.json"),
    "iteration_comparison": Path("evaluation/iteration_comparison_results.json"),
}


class SearchRequest(BaseModel):
    log: str = Field(min_length=1)
    top_k: int = Field(default=10, ge=1, le=50)


class SearchResult(BaseModel):
    id: str
    title: str
    url: str | None
    source: str
    source_type: str
    published_at: str | None
    score: float
    rank: int
    snippet: str | None
    method: str


class SearchResponse(BaseModel):
    mode: str
    results: list[SearchResult]


class MetricsResponse(BaseModel):
    pipeline: dict[str, Any]
    files: dict[str, Any]


def _search(mode: SearchMode, request: SearchRequest) -> SearchResponse:
    results = [SearchResult(**hit.__dict__) for hit in _SEARCH_SERVICE.search(request.log, mode=mode, top_k=request.top_k)]
    return SearchResponse(mode=mode, results=results)


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _pipeline_summary() -> dict[str, Any]:
    embedding_backend = os.environ.get("EMBEDDING_BACKEND", "auto")
    embedding_model = os.environ.get("EMBEDDING_MODEL", "intfloat/e5-small-v2")
    return {
        "stages": [
            "incident extraction",
            "query rewrite",
            "bm25 lexical retrieval",
            "learned dense pgvector retrieval",
            "weighted fusion",
            "heuristic scoring",
            "DeepSeek rerank",
        ],
        "embeddings": {
            "backend": embedding_backend,
            "model": embedding_model,
            "quantization": os.environ.get("EMBEDDING_QUANTIZATION", "none"),
            "dimensions": settings.embedding_dim,
            "query_prefix": os.environ.get("EMBEDDING_QUERY_PREFIX", "query: "),
            "document_prefix": os.environ.get("EMBEDDING_DOCUMENT_PREFIX", "passage: "),
        },
        "fusion": {
            "mode": os.environ.get("RETRIEVAL_FUSION_MODE", "rrf"),
        },
        "reranker": {
            "provider": "DeepSeek",
            "model": os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash"),
            "enabled": str(os.environ.get("DEEPSEEK_RERANK_ENABLED", "")).strip().lower() in {"1", "true", "yes", "on"},
            "top_n": int(os.environ.get("DEEPSEEK_RERANK_TOP_N", "12")),
        },
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/metrics", response_model=MetricsResponse)
def metrics() -> MetricsResponse:
    return MetricsResponse(
        pipeline=_pipeline_summary(),
        files={name: _load_json(path) for name, path in _METRIC_FILES.items()},
    )


@app.post("/search/bm25")
def search_bm25(request: SearchRequest) -> SearchResponse:
    return _search("bm25", request)


@app.post("/search/dense")
def search_dense(request: SearchRequest) -> SearchResponse:
    return _search("dense", request)


@app.post("/search/hybrid")
def search_hybrid(request: SearchRequest) -> SearchResponse:
    return _search("hybrid", request)


@app.post("/search/pgvector")
def search_pgvector(request: SearchRequest) -> SearchResponse:
    return _search("pgvector", request)
