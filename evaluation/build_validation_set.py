from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import psycopg
from psycopg.rows import dict_row

from database.migrate import _postgres_url
from database.settings import settings


DEFAULT_OUTPUT = Path("evaluation/data/validation_set.json")


STATUSPAGE_SQL = """
SELECT
    rl.id::text AS log_id,
    rl.dataset,
    rl.source AS log_source,
    rl.message AS log_message,
    rl.event_time,
    rn.id::text AS news_id,
    rn.source_type,
    rn.source AS news_source,
    rn.title,
    rn.url,
    rn.published_at,
    rn.body,
    'same_statuspage_incident_id' AS relevance_reason
FROM raw_logs rl
JOIN raw_news rn
  ON rn.source_type = 'statuspage_incident'
 AND rn.raw_payload->>'incident_id' = rl.raw_payload->>'incident_id'
WHERE rl.dataset = 'statuspage_incidents'
ORDER BY rl.event_time DESC NULLS LAST, rl.source
LIMIT %(limit)s
"""


OSV_SQL = """
SELECT
    rl.id::text AS log_id,
    rl.dataset,
    rl.source AS log_source,
    rl.message AS log_message,
    rl.event_time,
    rn.id::text AS news_id,
    rn.source_type,
    rn.source AS news_source,
    rn.title,
    rn.url,
    rn.published_at,
    rn.body,
    'same_osv_advisory_id' AS relevance_reason
FROM raw_logs rl
JOIN raw_news rn
  ON rn.source_type = 'osv_advisory'
 AND rn.raw_payload->>'advisory_id' = rl.raw_payload->>'advisory_id'
WHERE rl.dataset = 'osv_advisories'
ORDER BY rl.event_time DESC NULLS LAST, rl.source
LIMIT %(limit)s
"""


GITHUB_SQL = """
SELECT
    rl.id::text AS log_id,
    rl.dataset,
    rl.source AS log_source,
    rl.message AS log_message,
    rl.event_time,
    rn.id::text AS news_id,
    rn.source_type,
    rn.source AS news_source,
    rn.title,
    rn.url,
    rn.published_at,
    rn.body,
    'same_github_repo_within_one_hour' AS relevance_reason
FROM raw_logs rl
JOIN raw_news rn
  ON rn.source_type = 'github_release'
 AND rn.source = rl.source
 AND rn.published_at BETWEEN rl.event_time - interval '1 hour'
                         AND rl.event_time + interval '1 hour'
WHERE rl.dataset = 'gharchive_open_source'
ORDER BY rl.event_time DESC NULLS LAST, rl.source
LIMIT %(limit)s
"""


NEGATIVE_SQL = """
SELECT
    rn.id::text AS news_id,
    rn.source_type,
    rn.source,
    rn.title,
    rn.url,
    rn.published_at
FROM raw_news rn
WHERE rn.source_type = %(source_type)s
  AND rn.id::text <> %(positive_id)s
  AND (
      rn.source <> %(source)s
      OR rn.published_at IS DISTINCT FROM %(published_at)s
  )
ORDER BY abs(extract(epoch FROM (coalesce(rn.published_at, now()) - coalesce(%(published_at)s::timestamptz, now()))))
LIMIT %(limit)s
"""


def fetch_rows(limit_per_source: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with psycopg.connect(_postgres_url(settings.database_url), row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            for query in (STATUSPAGE_SQL, OSV_SQL, GITHUB_SQL):
                cur.execute(query, {"limit": limit_per_source})
                rows.extend(cur.fetchall())
    return rows


def fetch_negatives(row: dict[str, Any], limit: int) -> list[dict[str, Any]]:
    if limit <= 0:
        return []
    with psycopg.connect(_postgres_url(settings.database_url), row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute(
                NEGATIVE_SQL,
                {
                    "source_type": row["source_type"],
                    "positive_id": row["news_id"],
                    "source": row["news_source"],
                    "published_at": row["published_at"],
                    "limit": limit,
                },
            )
            return cur.fetchall()


def build_examples(rows: list[dict[str, Any]], negatives_per_example: int = 3) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for row in rows:
        negatives = fetch_negatives(row, negatives_per_example)
        examples.append(
            {
                "id": f"{row['dataset']}:{row['log_id']}:{row['news_id']}",
                "query": {
                    "log_id": row["log_id"],
                    "dataset": row["dataset"],
                    "source": row["log_source"],
                    "message": row["log_message"],
                    "event_time": row["event_time"].isoformat() if row["event_time"] else None,
                },
                "relevant_news": [
                    {
                        "news_id": row["news_id"],
                        "source_type": row["source_type"],
                        "source": row["news_source"],
                        "title": row["title"],
                        "url": row["url"],
                        "published_at": row["published_at"].isoformat() if row["published_at"] else None,
                        "relevance_reason": row["relevance_reason"],
                    }
                ],
                "negative_news": [
                    {
                        "news_id": item["news_id"],
                        "source_type": item["source_type"],
                        "source": item["source"],
                        "title": item["title"],
                        "url": item["url"],
                        "published_at": item["published_at"].isoformat() if item["published_at"] else None,
                        "negative_reason": "same_source_type_different_linkage",
                    }
                    for item in negatives
                ],
            }
        )
    return examples


def write_validation_set(output_path: Path, examples: list[dict[str, Any]]) -> None:
    payload = {
        "version": 1,
        "description": "Validation set for incident/status logs and relevant news records.",
        "labeling_rules": [
            "Statuspage update logs are relevant to statuspage incident reports with the same incident_id.",
            "OSV package log records are relevant to OSV advisory news with the same advisory_id.",
            "GH Archive activity logs are relevant to GitHub release news for the same repository within +/- 1 hour.",
            "Negative examples use the same source_type but intentionally different linkage.",
        ],
        "examples": examples,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a validation set from loaded raw_logs/raw_news pairs.")
    parser.add_argument("--limit-per-source", type=int, default=50)
    parser.add_argument("--negatives-per-example", type=int, default=3)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    examples = build_examples(fetch_rows(args.limit_per_source), args.negatives_per_example)
    write_validation_set(args.output, examples)
    print(f"validation_examples: {len(examples)}")
    print(f"output: {args.output}")


if __name__ == "__main__":
    main()
