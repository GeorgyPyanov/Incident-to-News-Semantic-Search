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
            cur.execute(
                """
                SELECT
                    count(*) FILTER (WHERE embedding IS NOT NULL)::bigint,
                    count(*)::bigint
                FROM raw_news
                """
            )
            embedded_raw_news, total_raw_news = cur.fetchone()
        print(f"total_objects: {total_objects}")
        embedded_raw_news = int(embedded_raw_news)
        total_raw_news = int(total_raw_news)
        unembedded_raw_news = max(0, total_raw_news - embedded_raw_news)
        coverage = (embedded_raw_news / total_raw_news * 100.0) if total_raw_news else 0.0
        print(f"raw_news_embeddings: {embedded_raw_news} / {total_raw_news}")
        print(f"raw_news_unembedded: {unembedded_raw_news}")
        print(f"raw_news_embedding_coverage_percent: {coverage:.2f}")
        if total_objects < min_total:
            raise SystemExit(f"Database has fewer than {min_total} objects.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check loaded database coverage.")
    parser.add_argument("--min-total", type=int, default=50_000)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    check_database(args.min_total)
