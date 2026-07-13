from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel, Field

from retrieval.db_search import DbNewsSearchService, SearchMode


app = FastAPI(title="Incident-to-News Semantic Search")


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


def _search(mode: SearchMode, request: SearchRequest) -> SearchResponse:
    service = DbNewsSearchService()
    results = [SearchResult(**hit.__dict__) for hit in service.search(request.log, mode=mode, top_k=request.top_k)]
    return SearchResponse(mode=mode, results=results)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


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
