from pathlib import Path

import psycopg

from database.settings import settings

MIGRATIONS_DIR = Path(__file__).parent / "migrations"


def _postgres_url(database_url: str) -> str:
    return database_url.replace("postgresql+asyncpg://", "postgresql://", 1)


def _asyncpg_url(database_url: str) -> str:
    return _postgres_url(database_url)


def apply_migrations() -> None:
    with psycopg.connect(_postgres_url(settings.database_url)) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    filename TEXT PRIMARY KEY,
                    applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            )
            cur.execute("SELECT filename FROM schema_migrations")
            applied = {row[0] for row in cur.fetchall()}

            for migration_path in sorted(MIGRATIONS_DIR.glob("*.sql")):
                if migration_path.name in applied:
                    continue

                sql = migration_path.read_text(encoding="utf-8")
                cur.execute(sql)
                cur.execute(
                    "INSERT INTO schema_migrations (filename) VALUES (%s)",
                    (migration_path.name,),
                )
                print(f"Applied migration: {migration_path.name}")


if __name__ == "__main__":
    apply_migrations()
