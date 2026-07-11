"""SQL fragments for dense, sparse, and hybrid retrieval."""

from __future__ import annotations

from collections.abc import Sequence


def _coalesce_expression(columns: Sequence[str]) -> str:
    # Build one snippet expression from whichever text columns the table exposes.
    if not columns:
        return "''"
    return "coalesce(" + ", ".join(columns) + ", '')"


def dense_search_sql(
    table: str,
    *,
    vector_column: str = "embedding",
    id_column: str = "id",
    snippet_columns: Sequence[str] = ("title", "description", "original_log", "content"),
) -> str:
    return f"""
SELECT
    {id_column} AS id,
    source_id,
    title,
    1 - ({vector_column} <=> %(query_embedding)s::vector) AS score,
    row_number() OVER (ORDER BY {vector_column} <=> %(query_embedding)s::vector) AS rank,
    left({_coalesce_expression(snippet_columns)}, 240) AS snippet
FROM {table}
WHERE {vector_column} IS NOT NULL
ORDER BY {vector_column} <=> %(query_embedding)s::vector
LIMIT %(limit)s
""".strip()


def full_text_search_sql(
    table: str,
    *,
    tsvector_column: str = "search_tsv",
    id_column: str = "id",
    query_config: str = "english",
    snippet_columns: Sequence[str] = ("title", "description", "original_log", "content"),
) -> str:
    return f"""
SELECT
    {id_column} AS id,
    source_id,
    title,
    ts_rank_cd({tsvector_column}, plainto_tsquery('{query_config}', %(query_text)s)) AS score,
    row_number() OVER (ORDER BY ts_rank_cd({tsvector_column}, plainto_tsquery('{query_config}', %(query_text)s)) DESC) AS rank,
    left({_coalesce_expression(snippet_columns)}, 240) AS snippet
FROM {table}
WHERE {tsvector_column} @@ plainto_tsquery('{query_config}', %(query_text)s)
ORDER BY score DESC
LIMIT %(limit)s
""".strip()

