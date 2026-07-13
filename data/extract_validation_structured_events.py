from __future__ import annotations

import argparse
import json
from pathlib import Path

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from database.migrate import _postgres_url
from database.settings import settings
from event_extraction.news_events import NewsStructuredEventExtractor


DEFAULT_VALIDATION_SET = Path("evaluation/data/validation_set.json")

SELECT_SQL = """
SELECT *
FROM raw_news
WHERE id::text = ANY(%(ids)s)
  AND NOT EXISTS (
      SELECT 1
      FROM structured_events se
      WHERE se.raw_news_id = raw_news.id
  )
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


def extract_for_validation(validation_path: Path) -> int:
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    news_ids = sorted(
        {
            item["news_id"]
            for example in validation["examples"]
            for item in example.get("relevant_news", [])
        }
    )
    extractor = NewsStructuredEventExtractor()
    inserted = 0
    with psycopg.connect(_postgres_url(settings.database_url), row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute(SELECT_SQL, {"ids": news_ids})
            for row in cur.fetchall():
                event = extractor.extract(dict(row))
                payload = event.model_dump()
                payload["metadata"] = Jsonb(payload.pop("metadata"))
                cur.execute(INSERT_SQL, payload)
                inserted += 1
        conn.commit()
    return inserted


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract structured events for validation-set relevant news.")
    parser.add_argument("--validation-set", type=Path, default=DEFAULT_VALIDATION_SET)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"validation_structured_events_inserted: {extract_for_validation(args.validation_set)}")


if __name__ == "__main__":
    main()
