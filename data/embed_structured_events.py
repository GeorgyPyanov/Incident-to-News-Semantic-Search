from __future__ import annotations

import argparse
import os

import psycopg
from psycopg.rows import dict_row

from database.migrate import _postgres_url
from database.settings import settings
from retrieval.embeddings import (
    build_document_text,
    build_embedding_client,
    validate_embedding_dimension,
)

SELECT_SQL = """
SELECT
    se.id,
    se.event_type,
    se.provider,
    se.title,
    se.summary,
    rn.source,
    rn.source_type,
    rn.raw_payload
FROM structured_events se
LEFT JOIN raw_news rn ON rn.id = se.raw_news_id
WHERE (%(refresh)s OR se.embedding IS NULL)
ORDER BY se.created_at
"""

UPDATE_SQL = """
UPDATE structured_events
SET embedding = %(embedding)s::vector,
    embedding_model = %(model)s,
    updated_at = now()
WHERE id = %(event_id)s
"""


def embed_structured_events(limit: int | None = 1000, refresh: bool = False) -> int:
    updated = 0
    client = build_embedding_client(
        backend=os.getenv("EMBEDDING_BACKEND", "auto"),
        model=os.getenv("EMBEDDING_MODEL", "intfloat/e5-small-v2"),
        dimensions=settings.embedding_dim,
        quantization=os.getenv("EMBEDDING_QUANTIZATION", "none"),
    )
    with psycopg.connect(_postgres_url(settings.database_url), row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            sql = SELECT_SQL
            params: dict[str, object] = {"refresh": refresh}
            if limit is not None:
                sql += "\nLIMIT %(limit)s"
                params["limit"] = limit
            cur.execute(sql, params)
            rows = cur.fetchall()
            for row in rows:
                text = build_document_text(_event_text(row))
                vector = client.embed_text(text)
                validate_embedding_dimension(vector, settings.embedding_dim, client.model_name)
                cur.execute(
                    UPDATE_SQL,
                    {
                        "event_id": row["id"],
                        "embedding": _vector_literal(vector),
                        "model": client.model_name,
                    },
                )
                updated += 1
        conn.commit()
    return updated


def _event_text(row: dict) -> str:
    return " ".join(
        str(row.get(key) or "")
        for key in ("source", "source_type", "event_type", "provider", "title", "summary", "raw_payload")
    )


def _vector_literal(values: list[float]) -> str:
    return "[" + ",".join(f"{value:.8f}" for value in values) + "]"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Embed structured_events with the configured local embedding model.")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--all", action="store_true", help="Refresh all structured events instead of only a limited batch.")
    parser.add_argument("--refresh", action="store_true", help="Recompute existing embeddings instead of only filling NULLs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    limit = None if args.all else args.limit
    print(f"structured_events_embedded: {embed_structured_events(limit, refresh=args.refresh)}")


if __name__ == "__main__":
    main()
