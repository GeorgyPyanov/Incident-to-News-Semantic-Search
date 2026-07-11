"""Apply the PostgreSQL schema for incidents and news articles."""

from __future__ import annotations

from pathlib import Path

import psycopg


def apply_schema(conn: psycopg.Connection, schema_path: str | Path | None = None) -> None:
    # Load the schema from disk so migrations stay explicit and reviewable.
    path = Path(schema_path) if schema_path is not None else Path(__file__).with_name("schema.sql")
    sql = path.read_text(encoding="utf-8")
    with conn.cursor() as cursor:
        cursor.execute(sql)
    conn.commit()

