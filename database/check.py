import argparse

import psycopg

from database.migrate import _postgres_url
from database.settings import settings


COUNT_QUERIES = {
    "news_sources": "SELECT count(*) FROM news_sources",
    "raw_news": "SELECT count(*) FROM raw_news",
    "raw_logs": "SELECT count(*) FROM raw_logs",
    "incidents": "SELECT count(*) FROM incidents",
    "structured_events": "SELECT count(*) FROM structured_events",
    "retrieval_logs": "SELECT count(*) FROM retrieval_logs",
    "reasoning_results": "SELECT count(*) FROM reasoning_results",
}


def check_database(min_total: int = 50_000) -> None:
    with psycopg.connect(_postgres_url(settings.database_url)) as conn:
        total_objects = 0
        with conn.cursor() as cur:
            for table_name, query in COUNT_QUERIES.items():
                cur.execute(query)
                value = int(cur.fetchone()[0])
                total_objects += value
                print(f"{table_name}: {value}")
        print(f"total_objects: {total_objects}")
        if total_objects < min_total:
            raise SystemExit(f"Database has fewer than {min_total} objects.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check loaded database coverage.")
    parser.add_argument("--min-total", type=int, default=50_000)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    check_database(args.min_total)
