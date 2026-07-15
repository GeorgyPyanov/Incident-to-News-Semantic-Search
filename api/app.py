from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, TypeVar

from fastapi import FastAPI
from pydantic import BaseModel, Field

from database.settings import settings
from retrieval.db_search import DbNewsSearchService, SearchMode
from retrieval.rag_answer import RagAnswerService


app = FastAPI(title="Incident-to-News Semantic Search")
_SEARCH_SERVICE = DbNewsSearchService()
_ANSWER_SERVICE = RagAnswerService(_SEARCH_SERVICE)
_METRIC_FILES = {
    "linked_validation": Path("evaluation/validation_results.json"),
    "blind_validation": Path("evaluation/qrels_validation_results.json"),
    "embedding_analysis": Path("evaluation/embedding_analysis.json"),
    "benchmark_search": Path("evaluation/benchmark_results.json"),
    "benchmark_real": Path("evaluation/benchmark_real_results.json"),
    "iteration_comparison": Path("evaluation/iteration_comparison_results.json"),
}
ModelT = TypeVar("ModelT", bound=BaseModel)


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


class AnswerCitation(BaseModel):
    id: str
    title: str
    url: str | None
    source: str
    source_type: str
    published_at: str | None
    rank: int
    score: float


class AnswerResponse(BaseModel):
    status: str
    answer: str
    citations: list[AnswerCitation]
    retrieval_mode: str
    model: str
    abstention_reason: str | None


class MetricsResponse(BaseModel):
    pipeline: dict[str, Any]
    files: dict[str, Any]


def _search(mode: SearchMode, request: SearchRequest) -> SearchResponse:
    results = [SearchResult(**hit.__dict__) for hit in _SEARCH_SERVICE.search(request.log, mode=mode, top_k=request.top_k)]
    return SearchResponse(mode=mode, results=results)


def _model_from_attrs(model_type: type[ModelT], value: object) -> ModelT:
    return model_type(**{field_name: getattr(value, field_name) for field_name in model_type.model_fields})


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _load_metric_file(name: str, path: Path) -> dict[str, Any] | None:
    payload = _load_json(path)
    if name == "benchmark_real" and payload is not None:
        database = payload.get("database") if isinstance(payload, dict) else None
        if isinstance(database, dict) and database.get("table") != "raw_news":
            return {
                "stale": True,
                "reason": "benchmark_real must be rerun after moving pgvector retrieval to raw_news",
                "expected_table": "raw_news",
                "actual_table": database.get("table"),
                "rerun": "py -m evaluation.benchmark_real --benchmark-document-embeddings --embedding-sample-size 100",
            }
        embedding = payload.get("embedding") if isinstance(payload, dict) else None
        if isinstance(embedding, dict):
            embedded_documents = int(embedding.get("embedded_documents") or 0)
            total_documents = int(embedding.get("total_documents") or 0)
            coverage_ratio = embedded_documents / total_documents if total_documents else 0.0
            if coverage_ratio < 0.95:
                payload = {
                    **payload,
                    "coverage_warning": {
                        "reason": "raw_news embeddings are incomplete",
                        "embedded_documents": embedded_documents,
                        "total_documents": total_documents,
                        "unembedded_documents": max(0, total_documents - embedded_documents),
                        "coverage_ratio": coverage_ratio,
                        "rerun": "py -m data.embed_raw_news --all",
                    },
                }
    return payload


def _pipeline_summary() -> dict[str, Any]:
    embedding_backend = os.environ.get("EMBEDDING_BACKEND", "auto")
    embedding_model = os.environ.get("EMBEDDING_MODEL", "intfloat/e5-small-v2")
    answer_provider = os.environ.get("RAG_GENERATOR_PROVIDER", "deepseek").strip().lower()
    reranker = _reranker_config()
    if answer_provider in {"ollama", "local"}:
        answer_model = os.environ.get("OLLAMA_MODEL", "qwen2.5:3b")
        answer_base_url = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    else:
        answer_provider = "deepseek"
        answer_model = os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash")
        answer_base_url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    return {
        "stages": [
            "incident extraction",
            "query rewrite",
            "bm25 lexical retrieval",
            "learned dense pgvector retrieval over raw_news",
            "weighted fusion",
            "heuristic scoring",
            "DeepSeek rerank" if reranker["enabled"] else "LLM rerank disabled",
            f"{answer_provider} answer generation with abstention",
        ],
        "embeddings": {
            "backend": embedding_backend,
            "model": embedding_model,
            "quantization": os.environ.get("EMBEDDING_QUANTIZATION", "none"),
            "dimensions": settings.embedding_dim,
            "query_prefix": os.environ.get("EMBEDDING_QUERY_PREFIX", "query: "),
            "document_prefix": os.environ.get("EMBEDDING_DOCUMENT_PREFIX", "passage: "),
            "corpus": {
                "table": "raw_news",
                "embedding_column": "embedding",
                "index": "ix_raw_news_embedding",
                "auxiliary_table": "structured_events",
            },
        },
        "fusion": {
            "mode": os.environ.get("RETRIEVAL_FUSION_MODE", "rrf"),
        },
        "reranker": reranker,
        "answer_generation": {
            "provider": answer_provider,
            "model": answer_model,
            "base_url": answer_base_url,
            "retrieval_mode": os.environ.get("RAG_RETRIEVAL_MODE", "hybrid"),
            "min_top_score": float(os.environ.get("RAG_MIN_TOP_SCORE", "0.35")),
            "min_supporting_documents": int(os.environ.get("RAG_MIN_SUPPORTING_DOCS", "1")),
            "min_evidence_overlap": float(os.environ.get("RAG_MIN_EVIDENCE_OVERLAP", "0.12")),
        },
    }


def _reranker_config() -> dict[str, Any]:
    return {
        "provider": "deepseek",
        "model": os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash"),
        "base_url": os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        "enabled": str(os.environ.get("DEEPSEEK_RERANK_ENABLED", "")).strip().lower() in {"1", "true", "yes", "on"},
        "top_n": int(os.environ.get("DEEPSEEK_RERANK_TOP_N", "12")),
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/metrics", response_model=MetricsResponse)
def metrics() -> MetricsResponse:
    return MetricsResponse(
        pipeline=_pipeline_summary(),
        files={name: _load_metric_file(name, path) for name, path in _METRIC_FILES.items()},
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


@app.post("/answer", response_model=AnswerResponse)
def answer(request: SearchRequest) -> AnswerResponse:
    result = _ANSWER_SERVICE.answer(request.log, top_k=request.top_k)
    return AnswerResponse(
        status=result.status,
        answer=result.answer,
        citations=[_model_from_attrs(AnswerCitation, citation) for citation in result.citations],
        retrieval_mode=result.retrieval_mode,
        model=result.model,
        abstention_reason=result.abstention_reason,
    )
