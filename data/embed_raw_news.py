from __future__ import annotations

import argparse
import json
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
    id,
    source,
    source_type,
    title,
    body,
    raw_region_hint,
    raw_payload,
    published_at,
    created_at
FROM raw_news
WHERE (%(refresh)s OR embedding IS NULL)
  AND (
      created_at,
      id
  ) > (
      COALESCE(%(last_created_at)s::timestamptz, '-infinity'::timestamptz),
      COALESCE(%(last_id)s::uuid, '00000000-0000-0000-0000-000000000000'::uuid)
  )
ORDER BY created_at, id
LIMIT %(limit)s
"""

UPDATE_SQL = """
UPDATE raw_news
SET embedding = %(embedding)s::vector,
    embedding_model = %(model)s
WHERE id = %(news_id)s
"""


def embed_raw_news(limit: int | None = 1000, refresh: bool = False, batch_size: int = 64) -> int:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if limit is not None and limit < 1:
        return 0

    updated = 0
    client = build_embedding_client(
        backend=os.getenv("EMBEDDING_BACKEND", "auto"),
        model=os.getenv("EMBEDDING_MODEL", "intfloat/e5-small-v2"),
        dimensions=settings.embedding_dim,
        quantization=os.getenv("EMBEDDING_QUANTIZATION", "none"),
    )
    with psycopg.connect(_postgres_url(settings.database_url), row_factory=dict_row) as conn:
        last_created_at = None
        last_id = None
        remaining = limit
        while remaining is None or remaining > 0:
            batch_limit = batch_size if remaining is None else min(batch_size, remaining)
            params: dict[str, object] = {
                "refresh": refresh,
                "last_created_at": last_created_at,
                "last_id": last_id,
                "limit": batch_limit,
            }
            with conn.cursor() as select_cur:
                select_cur.execute(SELECT_SQL, params)
                rows = select_cur.fetchall()
            if not rows:
                break

            texts = [build_document_text(raw_news_document_text(row)) for row in rows]
            vectors = client.embed_texts(texts)
            for vector in vectors:
                validate_embedding_dimension(vector, settings.embedding_dim, client.model_name)

            with conn.cursor() as update_cur:
                update_cur.executemany(
                    UPDATE_SQL,
                    [
                        {
                            "news_id": row["id"],
                            "embedding": _vector_literal(vector),
                            "model": client.model_name,
                        }
                        for row, vector in zip(rows, vectors, strict=True)
                    ],
                )
            conn.commit()
            updated += len(rows)
            last_created_at = rows[-1]["created_at"]
            last_id = rows[-1]["id"]
            if remaining is not None:
                remaining -= len(rows)
    return updated


def raw_news_document_text(row: dict) -> str:
    payload = row.get("raw_payload")
    if isinstance(payload, (dict, list)):
        payload_text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    else:
        payload_text = str(payload or "")
    return " ".join(
        str(part or "")
        for part in (
            row.get("source"),
            row.get("source_type"),
            row.get("title"),
            row.get("body"),
            row.get("raw_region_hint"),
            row.get("published_at"),
            payload_text,
        )
        if str(part or "").strip()
    )


def _vector_literal(values: list[float]) -> str:
    return "[" + ",".join(f"{value:.8f}" for value in values) + "]"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Embed raw_news with the configured local embedding model.")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--all", action="store_true", help="Process every raw_news row instead of only a limited batch.")
    parser.add_argument("--refresh", action="store_true", help="Recompute existing embeddings instead of only filling NULLs.")
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    limit = None if args.all else args.limit
    print(f"raw_news_embedded: {embed_raw_news(limit, refresh=args.refresh, batch_size=args.batch_size)}")


if __name__ == "__main__":
    main()
