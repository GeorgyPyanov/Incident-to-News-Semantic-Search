from __future__ import annotations

import os
import subprocess
import sys
import time

import psycopg

from database.migrate import _postgres_url, apply_migrations
from database.settings import settings


def main() -> None:
    wait_for_database()
    apply_migrations()
    command = [
        "python",
        "-m",
        "uvicorn",
        "api.app:app",
        "--host",
        os.environ.get("API_HOST", "0.0.0.0"),
        "--port",
        os.environ.get("API_PORT", "8000"),
    ]
    raise SystemExit(subprocess.call(command))


def wait_for_database(timeout_seconds: int = 120) -> None:
    deadline = time.monotonic() + int(os.environ.get("DATABASE_WAIT_TIMEOUT_SECONDS", timeout_seconds))
    last_error: Exception | None = None
    database_url = _postgres_url(settings.database_url)

    while time.monotonic() < deadline:
        try:
            with psycopg.connect(database_url) as connection:
                with connection.cursor() as cursor:
                    cursor.execute("SELECT 1")
                return
        except psycopg.OperationalError as error:
            last_error = error
            print("Waiting for PostgreSQL...", file=sys.stderr, flush=True)
            time.sleep(2)

    raise RuntimeError(f"PostgreSQL did not become ready: {last_error}") from last_error


if __name__ == "__main__":
    main()
