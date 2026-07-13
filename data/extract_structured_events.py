from __future__ import annotations

import argparse

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from database.migrate import _postgres_url
from database.settings import settings
from event_extraction.news_events import NewsStructuredEventExtractor


SELECT_SQL = """
SELECT rn.*
FROM raw_news rn
WHERE NOT EXISTS (
    SELECT 1
    FROM structured_events se
    WHERE se.raw_news_id = rn.id
)
ORDER BY rn.published_at DESC NULLS LAST, rn.fetched_at DESC
LIMIT %(limit)s
"""


INSERT_SQL = """
INSERT INTO structured_events (
    raw_news_id,
    event_type,
    provider,
    regions,
    title,
    summary,
    event_start,
    event_end,
    published_at,
    extraction_method,
    extraction_confidence,
    metadata
)
VALUES (
    %(raw_news_id)s,
    %(event_type)s,
    %(provider)s,
    %(regions)s,
    %(title)s,
    %(summary)s,
    %(event_start)s,
    %(event_end)s,
    %(published_at)s,
    %(extraction_method)s,
    %(extraction_confidence)s,
    %(metadata)s
)
"""


def extract_structured_events(limit: int) -> int:
    extractor = NewsStructuredEventExtractor()
    inserted = 0
    with psycopg.connect(_postgres_url(settings.database_url), row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute(SELECT_SQL, {"limit": limit})
            rows = cur.fetchall()
            for row in rows:
                event = extractor.extract(dict(row))
                payload = event.model_dump()
                payload["metadata"] = Jsonb(payload.pop("metadata"))
                cur.execute(INSERT_SQL, payload)
                inserted += 1
        conn.commit()
    return inserted


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract structured_events from raw_news.")
    parser.add_argument("--limit", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"structured_events_inserted: {extract_structured_events(args.limit)}")


if __name__ == "__main__":
    main()
