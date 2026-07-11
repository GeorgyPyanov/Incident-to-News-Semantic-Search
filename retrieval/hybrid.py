"""Reciprocal rank fusion helpers for dense and sparse retrieval."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

from .models import RankedCandidate, SearchHit


def rank_fusion(*ranked_lists: Sequence[RankedCandidate], k: float = 60.0) -> list[SearchHit]:
    # RRF combines ranked lists without assuming the score scales are comparable.
    scores: dict[tuple[str, int], float] = defaultdict(float)
    payloads: dict[tuple[str, int], dict] = {}
    titles: dict[tuple[str, int], str] = {}
    source_ids: dict[tuple[str, int], str | None] = {}

    for ranked in ranked_lists:
        for item in ranked:
            key = (item.source, item.id)
            scores[key] += 1.0 / (k + item.rank)
            payloads[key] = item.payload
            titles[key] = item.payload.get("title", "")
            source_ids[key] = item.payload.get("source_id")

    return [
        SearchHit(
            id=item_id,
            source_id=source_ids[(source, item_id)],
            title=titles[(source, item_id)],
            score=score,
            rank=rank,
            metadata={"source": source, **payloads[(source, item_id)]},
        )
        for rank, ((source, item_id), score) in enumerate(
            sorted(scores.items(), key=lambda pair: pair[1], reverse=True),
            start=1,
        )
    ]


def hybrid_rrf_sql(
    dense_query_sql: str,
    sparse_query_sql: str,
    limit: int = 10,
    k: float = 60.0,
) -> str:
    # The caller injects the subqueries; this wrapper performs the fusion.
    return f"""
WITH
dense_ranked AS (
    {dense_query_sql}
),
sparse_ranked AS (
    {sparse_query_sql}
),
unioned AS (
    SELECT
        coalesce(d.id, s.id) AS id,
        coalesce(d.source_id, s.source_id) AS source_id,
        coalesce(d.title, s.title) AS title,
        coalesce(d.score, 0) AS dense_score,
        coalesce(s.score, 0) AS sparse_score,
        coalesce(d.rank, 1000000) AS dense_rank,
        coalesce(s.rank, 1000000) AS sparse_rank,
        (1.0 / ({k} + coalesce(d.rank, 1000000)) + 1.0 / ({k} + coalesce(s.rank, 1000000))) AS rrf_score
    FROM dense_ranked d
    FULL OUTER JOIN sparse_ranked s USING (id, source_id, title)
)
SELECT *
FROM unioned
ORDER BY rrf_score DESC
LIMIT {limit}
""".strip()

