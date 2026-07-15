from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Literal

try:
    import psycopg
    from psycopg.rows import dict_row
except ImportError:  # pragma: no cover - exercised when optional DB deps are unavailable.
    psycopg = None  # type: ignore[assignment]
    dict_row = None  # type: ignore[assignment]

try:
    from database.migrate import _postgres_url
except ImportError:  # pragma: no cover - exercised when optional DB deps are unavailable.
    def _postgres_url(database_url: str) -> str:
        return database_url

from database.settings import settings
from retrieval.embeddings import (
    build_embedding_client,
    build_query_text,
    validate_embedding_dimension,
)
from retrieval.query_rewrite import rewrite_incident_query


SearchMode = Literal["bm25", "dense", "hybrid", "pgvector"]
TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_-]{2,}", re.IGNORECASE)
IDENTIFIER_RE = re.compile(r"\b(?:GHSA|CVE|RUSTSEC|PYSEC|GO)-[A-Za-z0-9_.-]+\b", re.IGNORECASE)


@dataclass(frozen=True)
class DbNewsHit:
    id: str
    title: str
    url: str | None
    source: str
    source_type: str
    published_at: str | None
    score: float
    rank: int
    snippet: str | None
    method: SearchMode


BM25_SQL = """
WITH q AS (
    SELECT plainto_tsquery('english', %(query)s) AS query
)
SELECT
    rn.id::text AS id,
    rn.title,
    rn.url,
    rn.source,
    rn.source_type,
    rn.published_at,
    ts_rank_cd(
        to_tsvector(
            'english',
            coalesce(rn.source, '') || ' ' ||
            coalesce(rn.title, '') || ' ' ||
            coalesce(rn.body, '') || ' ' ||
            rn.raw_payload::text
        ),
        q.query
    ) AS score,
    left(coalesce(rn.body, rn.title), 240) AS snippet
FROM raw_news rn, q
WHERE to_tsvector(
    'english',
    coalesce(rn.source, '') || ' ' ||
    coalesce(rn.title, '') || ' ' ||
    coalesce(rn.body, '') || ' ' ||
    rn.raw_payload::text
) @@ q.query
ORDER BY score DESC, rn.published_at DESC NULLS LAST
LIMIT %(limit)s
"""


IDENTIFIER_CANDIDATE_SQL = """
SELECT
    rn.id::text AS id,
    rn.title,
    rn.url,
    rn.source,
    rn.source_type,
    rn.published_at,
    rn.body,
    rn.raw_payload,
    left(coalesce(rn.body, rn.title), 240) AS snippet,
    1.0 AS lexical_score
FROM raw_news rn
WHERE (
    rn.source_type = 'osv_advisory'
    AND (
        rn.raw_payload->>'advisory_id' = %(identifier)s
        OR rn.raw_payload->'aliases' ? %(identifier)s
    )
)
   OR rn.title ILIKE %(pattern)s
   OR rn.source ILIKE %(pattern)s
ORDER BY rn.published_at DESC NULLS LAST
LIMIT %(limit)s
"""


PGVECTOR_SQL = """
SELECT
    rn.id::text AS id,
    rn.title,
    rn.url,
    rn.source,
    rn.source_type,
    rn.published_at,
    left(coalesce(rn.body, rn.title), 240) AS snippet,
    1 - (rn.embedding <=> %(embedding)s::vector) AS score
FROM raw_news rn
WHERE rn.embedding IS NOT NULL
ORDER BY rn.embedding <=> %(embedding)s::vector
LIMIT %(limit)s
"""


class DbNewsSearchService:
    def __init__(self) -> None:
        self._embedding_client = None
        self._query_embedding_cache: dict[str, list[float]] = {}
        self._migrations_applied = False
        self._hybrid_search = None

    def search(self, query: str, mode: SearchMode, top_k: int = 10) -> list[DbNewsHit]:
        top_k = max(1, min(top_k, 50))
        if mode == "bm25":
            return self.search_bm25(query, top_k)
        if mode == "dense":
            return self.search_dense(query, top_k)
        if mode == "pgvector":
            return self.search_pgvector(query, top_k)
        return self.search_hybrid(query, top_k)

    def search_bm25(self, query: str, top_k: int = 10) -> list[DbNewsHit]:
        if psycopg is None or dict_row is None:
            raise RuntimeError("Database search requires the optional psycopg dependency.")
        self._ensure_migrations()
        expanded_query = _expanded_query(query)
        with psycopg.connect(_postgres_url(settings.database_url), row_factory=dict_row) as conn:
            with conn.cursor() as cur:
                rows_by_id: dict[str, dict] = {}
                for row in self._identifier_rows(cur, query, limit=top_k):
                    rows_by_id[row["id"]] = {**row, "score": 10.0}
                cur.execute(BM25_SQL, {"query": expanded_query, "limit": top_k})
                for row in cur.fetchall():
                    rows_by_id.setdefault(row["id"], row)
        rows = sorted(rows_by_id.values(), key=lambda row: float(row.get("score") or 0.0), reverse=True)[:top_k]
        return [_row_to_hit(row, rank, "bm25") for rank, row in enumerate(rows, start=1)]

    def search_dense(self, query: str, top_k: int = 10, pool_size: int = 200) -> list[DbNewsHit]:
        return self._search_pgvector(query, top_k=top_k, method="dense")

    def search_pgvector(self, query: str, top_k: int = 10) -> list[DbNewsHit]:
        return self._search_pgvector(query, top_k=top_k, method="pgvector")

    def _search_pgvector(self, query: str, top_k: int, method: SearchMode) -> list[DbNewsHit]:
        if psycopg is None or dict_row is None:
            raise RuntimeError("Database search requires the optional psycopg dependency.")
        self._ensure_migrations()
        client = self._get_embedding_client()
        vector = self._query_vector(query)
        embedding = _vector_literal(vector)
        with psycopg.connect(_postgres_url(settings.database_url), row_factory=dict_row) as conn:
            with conn.cursor() as cur:
                cur.execute(PGVECTOR_SQL, {"embedding": embedding, "limit": top_k})
                rows = cur.fetchall()
        return [_row_to_hit(row, rank, method) for rank, row in enumerate(rows, start=1)]

    def search_hybrid(self, query: str, top_k: int = 10) -> list[DbNewsHit]:
        from retrieval.multistage import MultiStageNewsSearch

        self._ensure_migrations()
        if self._hybrid_search is None:
            self._hybrid_search = MultiStageNewsSearch(self)
        return self._hybrid_search.search(query, top_k=top_k)

    def close(self) -> None:
        hybrid_search = self._hybrid_search
        if hybrid_search is not None:
            hybrid_search.close()

    def _identifier_rows(self, cur, query: str, limit: int) -> list[dict]:
        rows: list[dict] = []
        seen: set[str] = set()
        for identifier in _identifiers(query):
            cur.execute(
                IDENTIFIER_CANDIDATE_SQL,
                {"identifier": identifier, "pattern": f"%{identifier}%", "limit": limit},
            )
            for row in cur.fetchall():
                if row["id"] not in seen:
                    seen.add(row["id"])
                    rows.append(row)
        return rows

    def _get_embedding_client(self):
        if self._embedding_client is None:
            self._embedding_client = build_embedding_client(
                backend=os.getenv("EMBEDDING_BACKEND", "auto"),
                model=os.getenv("EMBEDDING_MODEL", "intfloat/e5-small-v2"),
                dimensions=settings.embedding_dim,
                quantization=os.getenv("EMBEDDING_QUANTIZATION", "none"),
            )
        return self._embedding_client

    def _query_vector(self, query: str) -> list[float]:
        cached = self._query_embedding_cache.get(query)
        if cached is not None:
            return cached
        client = self._get_embedding_client()
        vector = client.embed_text(build_query_text(rewrite_incident_query(query)))
        validate_embedding_dimension(vector, settings.embedding_dim, client.model_name)
        self._query_embedding_cache[query] = vector
        return vector

    def _ensure_migrations(self) -> None:
        if self._migrations_applied:
            return
        from database.migrate import apply_migrations

        apply_migrations()
        self._migrations_applied = True


def _row_to_hit(row: dict, rank: int, method: SearchMode) -> DbNewsHit:
    published_at = row.get("published_at")
    return DbNewsHit(
        id=str(row["id"]),
        title=row.get("title") or "",
        url=row.get("url"),
        source=row.get("source") or "",
        source_type=row.get("source_type") or "",
        published_at=published_at.isoformat() if published_at else None,
        score=float(row.get("score") or 0.0),
        rank=rank,
        snippet=row.get("snippet"),
        method=method,
    )


def _tokens(text: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_RE.finditer(text or "")]


def _expanded_query(query: str) -> str:
    rewritten = rewrite_incident_query(query)
    if not rewritten.strip():
        return query
    if rewritten.strip().lower() == query.strip().lower():
        return query
    return f"{query} {rewritten}"


def _identifiers(text: str) -> list[str]:
    seen: set[str] = set()
    values: list[str] = []
    for match in IDENTIFIER_RE.finditer(text or ""):
        value = match.group(0)
        key = value.lower()
        if key not in seen:
            seen.add(key)
            values.append(value)
    return values


def _vector_literal(values: list[float]) -> str:
    return "[" + ",".join(f"{value:.8f}" for value in values) + "]"
