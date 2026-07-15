"""Command-line entry points for embeddings and retrieval experiments."""

from __future__ import annotations

import argparse
import json
import os

from dotenv import load_dotenv
import psycopg
from pgvector.psycopg import register_vector

from database.bootstrap import apply_schema
from event_extraction.formatting import build_incident_embedding_text, build_incident_search_text
from event_extraction.schemas import IncidentData
from .benchmark import compare_dense_strategies, run_dense_benchmark
from .embeddings import build_embedding_client
from .sql import dense_search_sql, full_text_search_sql


def _connect():
    # Register pgvector adapters on every connection before issuing vector queries.
    load_dotenv()
    conn = psycopg.connect(os.environ["DATABASE_URL"])
    register_vector(conn)
    return conn


def cmd_init_db(args: argparse.Namespace) -> None:
    conn = _connect()
    try:
        apply_schema(conn, args.schema)
        print("schema applied")
    finally:
        conn.close()


def cmd_embed_incident(args: argparse.Namespace) -> None:
    client = build_embedding_client(
        backend=os.getenv("EMBEDDING_BACKEND", "auto"),
        model=os.getenv("EMBEDDING_MODEL", "intfloat/e5-small-v2"),
        dimensions=int(os.getenv("EMBEDDING_DIM", "384")),
        quantization=os.getenv("EMBEDDING_QUANTIZATION", "none"),
    )
    incident = IncidentData(
        original_log=args.original_log,
        entities=tuple(args.entities or ()),
        locations=tuple(args.locations or ()),
        event_types=tuple(args.event_types or ()),
        dates=tuple(args.dates or ()),
        products=tuple(args.products or ()),
        services=tuple(args.services or ()),
        error_descriptions=tuple(args.error_descriptions or ()),
    )
    text = build_incident_embedding_text(incident)
    vector = client.embed_text(text)
    print(json.dumps({"text": text, "dimensions": len(vector), "embedding": vector[:8]}, ensure_ascii=False, indent=2))


def cmd_benchmark_dense(args: argparse.Namespace) -> None:
    conn = _connect()
    try:
        with conn.cursor() as cursor:
            # Run the same query twice: once with the normal planner and once with index usage disabled.
            params = {"query_embedding": json.loads(args.query_embedding), "limit": args.limit}
            indexed = run_dense_benchmark(
                cursor,
                dense_search_sql(args.table, snippet_columns=args.snippet_columns),
                params,
                label="indexed",
                explain=True,
            )
            cursor.execute("SET LOCAL enable_indexscan = off")
            cursor.execute("SET LOCAL enable_bitmapscan = off")
            cursor.execute("SET LOCAL enable_seqscan = on")
            seqscan = run_dense_benchmark(
                cursor,
                dense_search_sql(args.table, snippet_columns=args.snippet_columns),
                params,
                label="seqscan",
                explain=True,
            )
            print(json.dumps(compare_dense_strategies(indexed, seqscan), ensure_ascii=False, indent=2, default=str))
    finally:
        conn.close()


def cmd_fulltext_search(args: argparse.Namespace) -> None:
    conn = _connect()
    try:
        with conn.cursor() as cursor:
            # This path is the lexical baseline used for BM25-style comparison.
            cursor.execute(
                full_text_search_sql(
                    args.table,
                    tsvector_column=args.tsvector_column,
                    snippet_columns=args.snippet_columns,
                ),
                {"query_text": args.query_text, "limit": args.limit},
            )
            rows = cursor.fetchall()
            print(json.dumps(rows, ensure_ascii=False, indent=2, default=str))
    finally:
        conn.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="retrieval")
    sub = parser.add_subparsers(dest="command", required=True)

    init_db = sub.add_parser("init-db", help="Apply the PostgreSQL schema.")
    init_db.add_argument("--schema", help="Optional path to schema.sql")
    init_db.set_defaults(func=cmd_init_db)

    embed = sub.add_parser("embed-incident", help="Generate an embedding for an incident log.")
    embed.add_argument("--original-log", required=True)
    embed.add_argument("--entities", nargs="*")
    embed.add_argument("--locations", nargs="*")
    embed.add_argument("--event-types", nargs="*")
    embed.add_argument("--dates", nargs="*")
    embed.add_argument("--products", nargs="*")
    embed.add_argument("--services", nargs="*")
    embed.add_argument("--error-descriptions", nargs="*")
    embed.set_defaults(func=cmd_embed_incident)

    bench = sub.add_parser("benchmark-dense", help="Compare dense search with and without indexes.")
    bench.add_argument("--table", default="incidents")
    bench.add_argument("--query-embedding", required=True, help="JSON list with the query vector.")
    bench.add_argument("--limit", type=int, default=10)
    bench.add_argument("--snippet-columns", nargs="*", default=("title", "description", "original_log", "location"))
    bench.set_defaults(func=cmd_benchmark_dense)

    fts = sub.add_parser("fulltext-search", help="Run a BM25-style full-text search.")
    fts.add_argument("--table", default="incidents")
    fts.add_argument("--tsvector-column", default="search_tsv")
    fts.add_argument("--query-text", required=True)
    fts.add_argument("--limit", type=int, default=10)
    fts.add_argument("--snippet-columns", nargs="*", default=("title", "description", "original_log", "location"))
    fts.set_defaults(func=cmd_fulltext_search)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

