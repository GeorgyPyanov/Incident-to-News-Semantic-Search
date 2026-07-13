from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from typing import Literal

import psycopg
from psycopg.rows import dict_row

from database.migrate import _postgres_url
from database.settings import settings


SearchMode = Literal["bm25", "dense", "hybrid"]
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


CANDIDATE_SQL = """
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
    rn.body,
    rn.raw_payload,
    left(coalesce(rn.body, rn.title), 240) AS snippet,
    ts_rank_cd(
        to_tsvector(
            'english',
            coalesce(rn.source, '') || ' ' ||
            coalesce(rn.title, '') || ' ' ||
            coalesce(rn.body, '') || ' ' ||
            rn.raw_payload::text
        ),
        q.query
    ) AS lexical_score
FROM raw_news rn, q
WHERE to_tsvector(
    'english',
    coalesce(rn.source, '') || ' ' ||
    coalesce(rn.title, '') || ' ' ||
    coalesce(rn.body, '') || ' ' ||
    rn.raw_payload::text
) @@ q.query
ORDER BY lexical_score DESC, rn.published_at DESC NULLS LAST
LIMIT %(limit)s
"""


FALLBACK_CANDIDATE_SQL = """
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
    0.0 AS lexical_score
FROM raw_news rn
WHERE rn.title ILIKE %(pattern)s
   OR rn.body ILIKE %(pattern)s
   OR rn.source ILIKE %(pattern)s
   OR rn.raw_payload::text ILIKE %(pattern)s
ORDER BY rn.published_at DESC NULLS LAST
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


class DbNewsSearchService:
    def search(self, query: str, mode: SearchMode, top_k: int = 10) -> list[DbNewsHit]:
        top_k = max(1, min(top_k, 50))
        if mode == "bm25":
            return self.search_bm25(query, top_k)
        if mode == "dense":
            return self.search_dense(query, top_k)
        return self.search_hybrid(query, top_k)

    def search_bm25(self, query: str, top_k: int = 10) -> list[DbNewsHit]:
        with psycopg.connect(_postgres_url(settings.database_url), row_factory=dict_row) as conn:
            with conn.cursor() as cur:
                rows_by_id: dict[str, dict] = {}
                for row in self._identifier_rows(cur, query, limit=top_k):
                    rows_by_id[row["id"]] = {**row, "score": 10.0}
                cur.execute(BM25_SQL, {"query": query, "limit": top_k})
                for row in cur.fetchall():
                    rows_by_id.setdefault(row["id"], row)
        rows = sorted(rows_by_id.values(), key=lambda row: float(row.get("score") or 0.0), reverse=True)[:top_k]
        return [_row_to_hit(row, rank, "bm25") for rank, row in enumerate(rows, start=1)]

    def search_dense(self, query: str, top_k: int = 10, pool_size: int = 200) -> list[DbNewsHit]:
        candidates = self._candidate_rows(query, pool_size)
        query_vec = _hashed_embedding(query)
        scored = []
        for row in candidates:
            text = " ".join(
                str(row.get(key) or "") for key in ("source", "source_type", "title", "body", "raw_payload")
            )
            score = _cosine(query_vec, _hashed_embedding(text))
            scored.append((score, row))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            _row_to_hit({**row, "score": score}, rank, "dense")
            for rank, (score, row) in enumerate(scored[:top_k], start=1)
        ]

    def search_hybrid(self, query: str, top_k: int = 10) -> list[DbNewsHit]:
        bm25 = self.search_bm25(query, top_k=50)
        dense = self.search_dense(query, top_k=50)
        by_id: dict[str, DbNewsHit] = {}
        scores: dict[str, float] = {}
        for hits in (bm25, dense):
            for hit in hits:
                by_id.setdefault(hit.id, hit)
                scores[hit.id] = scores.get(hit.id, 0.0) + 1.0 / (60 + hit.rank)
        ranked_ids = sorted(scores, key=lambda item_id: scores[item_id], reverse=True)[:top_k]
        return [
            DbNewsHit(
                **{
                    **by_id[item_id].__dict__,
                    "score": scores[item_id],
                    "rank": rank,
                    "method": "hybrid",
                }
            )
            for rank, item_id in enumerate(ranked_ids, start=1)
        ]

    def _candidate_rows(self, query: str, limit: int) -> list[dict]:
        with psycopg.connect(_postgres_url(settings.database_url), row_factory=dict_row) as conn:
            with conn.cursor() as cur:
                rows_by_id: dict[str, dict] = {}
                for row in self._identifier_rows(cur, query, limit):
                    rows_by_id.setdefault(row["id"], row)
                cur.execute(CANDIDATE_SQL, {"query": query, "limit": limit})
                for row in cur.fetchall():
                    rows_by_id.setdefault(row["id"], row)
                rows = list(rows_by_id.values())
                if rows:
                    return rows[:limit]
                tokens = _tokens(query)
                pattern = f"%{tokens[0]}%" if tokens else "%"
                cur.execute(FALLBACK_CANDIDATE_SQL, {"pattern": pattern, "limit": limit})
                return list(cur.fetchall())

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


def _hashed_embedding(text: str, dimensions: int = 256) -> dict[int, float]:
    vector: dict[int, float] = {}
    for token in _tokens(text):
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        bucket = int.from_bytes(digest[:4], "big") % dimensions
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[bucket] = vector.get(bucket, 0.0) + sign
    norm = math.sqrt(sum(value * value for value in vector.values()))
    if not norm:
        return vector
    return {key: value / norm for key, value in vector.items()}


def _cosine(left: dict[int, float], right: dict[int, float]) -> float:
    if not left or not right:
        return 0.0
    if len(left) > len(right):
        left, right = right, left
    return sum(value * right.get(key, 0.0) for key, value in left.items())
