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


def check_database() -> None:
    with psycopg.connect(_postgres_url(settings.database_url)) as conn:
        total_objects = 0
        with conn.cursor() as cur:
            for table_name, query in COUNT_QUERIES.items():
                cur.execute(query)
                value = int(cur.fetchone()[0])
                total_objects += value
                print(f"{table_name}: {value}")
        print(f"total_objects: {total_objects}")
        if total_objects < 50_000:
            raise SystemExit("Database has fewer than 50 000 objects.")


if __name__ == "__main__":
    check_database()
