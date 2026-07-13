from __future__ import annotations

import argparse
import hashlib
import math
import re

import psycopg
from psycopg.rows import dict_row

from database.migrate import _postgres_url
from database.settings import settings


TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_-]{2,}", re.IGNORECASE)

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
WHERE se.embedding IS NULL
ORDER BY se.created_at
LIMIT %(limit)s
"""

UPDATE_SQL = """
UPDATE structured_events
SET embedding = %(embedding)s::vector,
    embedding_model = %(model)s,
    updated_at = now()
WHERE id = %(event_id)s
"""


def embed_structured_events(limit: int) -> int:
    updated = 0
    with psycopg.connect(_postgres_url(settings.database_url), row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute(SELECT_SQL, {"limit": limit})
            rows = cur.fetchall()
            for row in rows:
                text = _event_text(row)
                cur.execute(
                    UPDATE_SQL,
                    {
                        "event_id": row["id"],
                        "embedding": _vector_literal(_hashed_vector(text, settings.embedding_dim)),
                        "model": f"hashing-vectorizer-{settings.embedding_dim}",
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


def _hashed_vector(text: str, dimensions: int) -> list[float]:
    values = [0.0] * dimensions
    for token in TOKEN_RE.findall(text or ""):
        digest = hashlib.blake2b(token.lower().encode("utf-8"), digest_size=8).digest()
        bucket = int.from_bytes(digest[:4], "big") % dimensions
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        values[bucket] += sign
    norm = math.sqrt(sum(value * value for value in values))
    if norm:
        values = [value / norm for value in values]
    return values


def _vector_literal(values: list[float]) -> str:
    return "[" + ",".join(f"{value:.8f}" for value in values) + "]"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Embed structured_events with a local hashing vectorizer.")
    parser.add_argument("--limit", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"structured_events_embedded: {embed_structured_events(args.limit)}")


if __name__ == "__main__":
    main()
