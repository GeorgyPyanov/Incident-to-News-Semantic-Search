"""Benchmark the real PostgreSQL/pgvector retrieval path.

The default command is read-only: it uses existing structured-event embeddings
and the existing HNSW index. Expensive mutations require explicit CLI flags.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import time
from functools import lru_cache
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from evaluation.benchmark_search import collect_platform_info


DEFAULT_VALIDATION_SET = Path("evaluation/data/validation_set.json")
DEFAULT_OUTPUT = Path("evaluation/benchmark_real_results.json")
DEFAULT_SEARCHES = 100
DEFAULT_WARMUPS = 1
DEFAULT_TOP_K = 10
DEFAULT_EMBEDDING_LIMIT = 1000
DEFAULT_EMBEDDING_SAMPLE_SIZE = 100
EMBEDDING_BATCH_SIZE = 1
INDEX_NAME = "ix_structured_events_embedding"
TABLE_NAME = "structured_events"


class Cursor(Protocol):
    def execute(self, query: str, params: dict[str, Any] | None = None) -> Any: ...
    def fetchone(self) -> Any: ...
    def fetchall(self) -> list[Any]: ...


class Connection(Protocol):
    def cursor(self) -> Any: ...
    def commit(self) -> None: ...


LatencyClock = Callable[[], float]
ConnectionFactory = Callable[[], Any]
QueryEmbedder = Callable[[str, int], str]
EmbeddingGenerator = Callable[[int], int]
DocumentSampleLoader = Callable[[Cursor, int], list[dict[str, Any]]]
DocumentEmbedder = Callable[[dict[str, Any], int], list[float]]


EMBEDDING_METADATA_SQL = """
SELECT
    count(*) FILTER (WHERE embedding IS NOT NULL)::bigint AS embedded_documents,
    count(*)::bigint AS total_documents,
    min(vector_dims(embedding)) FILTER (WHERE embedding IS NOT NULL) AS vector_dimension,
    array_remove(array_agg(DISTINCT embedding_model), NULL) AS embedding_models
FROM structured_events
"""


INDEX_METADATA_SQL = """
SELECT
    index_class.relname AS index_name,
    access_method.amname AS index_type,
    pg_relation_size(index_class.oid)::bigint AS size_on_disk_bytes,
    pg_total_relation_size(index_class.oid)::bigint AS total_size_on_disk_bytes,
    pg_get_indexdef(index_class.oid) AS definition,
    index_class.reloptions AS reloptions
FROM pg_class AS index_class
JOIN pg_index AS index_info ON index_info.indexrelid = index_class.oid
JOIN pg_class AS table_class ON table_class.oid = index_info.indrelid
JOIN pg_am AS access_method ON access_method.oid = index_class.relam
WHERE table_class.relname = 'structured_events'
  AND index_class.relname = 'ix_structured_events_embedding'
"""


DOCUMENT_SAMPLE_SQL = """
/* benchmark_document_sample: read-only input for the production embedder */
SELECT
    se.id,
    se.event_type,
    se.provider,
    se.title,
    se.summary,
    rn.source,
    rn.source_type,
    rn.raw_payload
FROM structured_events AS se
LEFT JOIN raw_news AS rn ON rn.id = se.raw_news_id
WHERE se.embedding IS NOT NULL
ORDER BY md5(se.id::text)
LIMIT %(limit)s
"""


def run_real_benchmark(
    *,
    validation_path: str | Path = DEFAULT_VALIDATION_SET,
    output_path: str | Path = DEFAULT_OUTPUT,
    top_k: int = DEFAULT_TOP_K,
    searches: int = DEFAULT_SEARCHES,
    warmups: int = DEFAULT_WARMUPS,
    generate_embeddings: bool = False,
    embedding_limit: int = DEFAULT_EMBEDDING_LIMIT,
    rebuild_index: bool = False,
    benchmark_document_embeddings: bool = False,
    embedding_sample_size: int = DEFAULT_EMBEDDING_SAMPLE_SIZE,
    connection_factory: ConnectionFactory | None = None,
    query_embedder: QueryEmbedder | None = None,
    embedding_generator: EmbeddingGenerator | None = None,
    document_sample_loader: DocumentSampleLoader | None = None,
    document_embedder: DocumentEmbedder | None = None,
    search_sql: str | None = None,
    hardware: dict[str, Any] | None = None,
    timestamp: str | None = None,
    clock: LatencyClock = time.perf_counter,
    save: bool = True,
) -> dict[str, Any]:
    """Run a real-system benchmark; mutation is opt-in only."""

    if searches < 100:
        raise ValueError("searches must be at least 100")
    if warmups < 1:
        raise ValueError("warmups must be at least 1")
    if top_k < 1:
        raise ValueError("top_k must be positive")
    if embedding_limit < 1:
        raise ValueError("embedding_limit must be positive")
    if embedding_sample_size < 1:
        raise ValueError("embedding_sample_size must be positive")
    if benchmark_document_embeddings and (generate_embeddings or rebuild_index):
        raise ValueError(
            "--benchmark-document-embeddings cannot be combined with database mutation flags"
        )

    queries = load_validation_queries(validation_path)
    if not queries:
        raise ValueError("validation set must contain at least one query")

    factory = connection_factory or default_connection_factory
    embed_query = query_embedder or default_query_embedder
    sql = search_sql or default_search_sql()
    benchmark_hardware = hardware or collect_platform_info()

    generation = {
        "requested": generate_embeddings,
        "documents_generated": 0,
        "batch_size": EMBEDDING_BATCH_SIZE,
        "total_generation_time_seconds": None,
        "average_time_per_document_seconds": None,
        "throughput_documents_per_second": None,
    }
    if generate_embeddings:
        generator = embedding_generator or default_embedding_generator
        started = clock()
        generated = generator(embedding_limit)
        elapsed = clock() - started
        generation.update(
            {
                "documents_generated": generated,
                "total_generation_time_seconds": elapsed,
                "average_time_per_document_seconds": elapsed / generated if generated else None,
                "throughput_documents_per_second": generated / elapsed if elapsed > 0 else None,
            }
        )

    with factory() as connection:
        with connection.cursor() as cursor:
            # REINDEX is the only mutation allowed on this connection. Otherwise
            # PostgreSQL itself enforces the benchmark's read-only promise.
            if not rebuild_index:
                cursor.execute("SET TRANSACTION READ ONLY")

            index_build_seconds: float | None = None
            if rebuild_index:
                started = clock()
                cursor.execute(f"REINDEX INDEX {INDEX_NAME}")
                connection.commit()
                index_build_seconds = clock() - started

            embedding_metadata = get_embedding_metadata(cursor)
            index_metadata = get_index_metadata(cursor)
            dimension = int(embedding_metadata["vector_dimension"] or _configured_dimension())

            document_embedding_result: dict[str, Any] | None = None
            if benchmark_document_embeddings:
                load_sample = document_sample_loader or default_document_sample_loader
                embed_document = document_embedder or default_document_embedder
                sample = load_sample(cursor, embedding_sample_size)
                document_embedding_result = benchmark_document_sample(
                    sample,
                    dimension=dimension,
                    embedder=embed_document,
                    hardware=benchmark_hardware,
                    clock=clock,
                )

            for warmup_number in range(warmups):
                query = queries[warmup_number % len(queries)]
                query_vector = embed_query(query, dimension)
                cursor.execute(sql, {"embedding": query_vector, "limit": top_k})
                cursor.fetchall()

            query_embedding_latencies: list[float] = []
            index_search_latencies: list[float] = []
            end_to_end_latencies: list[float] = []
            for search_number in range(searches):
                query = queries[search_number % len(queries)]
                retrieval_started = clock()

                embedding_started = clock()
                query_vector = embed_query(query, dimension)
                query_embedding_latencies.append((clock() - embedding_started) * 1000.0)

                database_started = clock()
                cursor.execute(sql, {"embedding": query_vector, "limit": top_k})
                cursor.fetchall()
                index_search_latencies.append((clock() - database_started) * 1000.0)

                end_to_end_latencies.append((clock() - retrieval_started) * 1000.0)

            explain = explain_index_usage(cursor, sql, queries[0], dimension, top_k, embed_query)

    models = list(embedding_metadata.get("embedding_models") or [])
    result: dict[str, Any] = {
        "benchmark_type": "real_postgresql_pgvector",
        "timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
        "platform": benchmark_hardware,
        "database": {
            "table": TABLE_NAME,
            "read_only_default": True,
            "read_only_run": not generate_embeddings and not rebuild_index,
        },
        "dataset": {
            "validation_set": str(validation_path),
            "available_queries": len(queries),
            "measured_query_executions": searches,
        },
        "embedding": {
            "provider": "local",
            "model": models[0] if len(models) == 1 else models,
            "models_present": models,
            "vector_dimension": dimension,
            "embedded_documents": int(embedding_metadata["embedded_documents"]),
            "total_documents": int(embedding_metadata["total_documents"]),
            **generation,
        },
        "index": {
            "name": index_metadata["index_name"],
            "type": index_metadata["index_type"],
            "configuration": parse_index_configuration(index_metadata),
            "definition": index_metadata["definition"],
            "size_on_disk_bytes": int(index_metadata["size_on_disk_bytes"]),
            "total_size_on_disk_bytes": int(index_metadata["total_size_on_disk_bytes"]),
            "rebuild_requested": rebuild_index,
            "build_time_seconds": index_build_seconds,
        },
        "search": {
            "top_k": top_k,
            "warmup_searches": warmups,
            "measured_searches": searches,
            "clock": "time.perf_counter",
            "query_embedding_latency_ms": latency_statistics(query_embedding_latencies),
            "provider_api_latency_ms": None,
            "database_index_search_latency_ms": latency_statistics(index_search_latencies),
            "end_to_end_retrieval_latency_ms": latency_statistics(end_to_end_latencies),
        },
        "explain_analyze": explain,
    }
    if document_embedding_result is not None:
        result["document_embedding_benchmark"] = document_embedding_result

    if save:
        save_results(result, output_path)
    return result


def load_validation_queries(path: str | Path) -> tuple[str, ...]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return tuple(
        str(example.get("query", {}).get("message") or "").strip()
        for example in payload.get("examples", [])
        if str(example.get("query", {}).get("message") or "").strip()
    )


def get_embedding_metadata(cursor: Cursor) -> dict[str, Any]:
    cursor.execute(EMBEDDING_METADATA_SQL)
    row = cursor.fetchone()
    if not row or int(_row_value(row, "embedded_documents", 0) or 0) == 0:
        raise RuntimeError(
            "No existing structured-event embeddings were found. "
            "Run with --generate-embeddings only if database modification is intended."
        )
    return {
        "embedded_documents": _row_value(row, "embedded_documents", 0),
        "total_documents": _row_value(row, "total_documents", 1),
        "vector_dimension": _row_value(row, "vector_dimension", 2),
        "embedding_models": _row_value(row, "embedding_models", 3) or [],
    }


def get_index_metadata(cursor: Cursor) -> dict[str, Any]:
    cursor.execute(INDEX_METADATA_SQL)
    row = cursor.fetchone()
    if not row:
        raise RuntimeError(
            f"Required pgvector index {INDEX_NAME!r} was not found. "
            "Apply the existing database migrations before benchmarking."
        )
    return {
        "index_name": _row_value(row, "index_name", 0),
        "index_type": _row_value(row, "index_type", 1),
        "size_on_disk_bytes": _row_value(row, "size_on_disk_bytes", 2),
        "total_size_on_disk_bytes": _row_value(row, "total_size_on_disk_bytes", 3),
        "definition": _row_value(row, "definition", 4),
        "reloptions": _row_value(row, "reloptions", 5) or [],
    }


def default_document_sample_loader(cursor: Cursor, sample_size: int) -> list[dict[str, Any]]:
    cursor.execute(DOCUMENT_SAMPLE_SQL, {"limit": sample_size})
    rows = cursor.fetchall()
    if not rows:
        raise RuntimeError("No existing embedded structured events were available for sampling")
    return [dict(row) for row in rows]


def default_document_embedder(row: dict[str, Any], dimension: int) -> list[float]:
    # These are deliberately the exact production text preparation and
    # embedding functions used by data.embed_structured_events.
    try:
        from data.embed_structured_events import _event_text
        from retrieval.embeddings import build_document_text, validate_embedding_dimension
    except ImportError as error:
        raise RuntimeError("Document embedding benchmarking requires requirements.txt") from error
    vector = _benchmark_embedding_client(dimension).embed_text(build_document_text(_event_text(row)))
    validate_embedding_dimension(vector, dimension, _benchmark_embedding_client(dimension).model_name)
    return vector


def benchmark_document_sample(
    sample: Sequence[dict[str, Any]],
    *,
    dimension: int,
    embedder: DocumentEmbedder,
    hardware: dict[str, Any],
    clock: LatencyClock = time.perf_counter,
) -> dict[str, Any]:
    if not sample:
        raise ValueError("document embedding sample must not be empty")

    # Initialize/import the generator and warm its code path before timing.
    embedder(sample[0], dimension)

    per_document_ms: list[float] = []
    total_started = clock()
    for row in sample:
        document_started = clock()
        vector = embedder(row, dimension)
        per_document_ms.append((clock() - document_started) * 1000.0)
        if len(vector) != dimension:
            raise RuntimeError(
                f"Production embedder returned dimension {len(vector)}; expected {dimension}"
            )
    total_seconds = clock() - total_started

    return {
        "provider": "local",
        "model": _benchmark_embedding_client(dimension).model_name,
        "vector_dimension": dimension,
        "sample_size": len(sample),
        "batch_size": EMBEDDING_BATCH_SIZE,
        "warmup_documents": 1,
        "clock": "time.perf_counter",
        "total_generation_time_seconds": total_seconds,
        "per_document_latency_ms": latency_statistics(per_document_ms),
        "throughput_documents_per_second": len(sample) / total_seconds if total_seconds > 0 else None,
        "gpu_used": False,
        "hardware": {
            "cpu": hardware["cpu"],
            "total_ram_bytes": hardware["total_ram_bytes"],
            "gpu": hardware["gpu"],
        },
        "database_writes_performed": False,
        "embeddings_persisted": False,
        "index_modified": False,
        "source_functions": [
            "data.embed_structured_events._event_text",
            "retrieval.embeddings.build_document_text",
            "retrieval.embeddings.build_embedding_client",
        ],
    }


def parse_index_configuration(metadata: dict[str, Any]) -> dict[str, Any]:
    configuration: dict[str, Any] = {}
    for option in metadata.get("reloptions") or []:
        key, _, value = str(option).partition("=")
        configuration[key] = _numeric_value(value)
    definition = str(metadata.get("definition") or "")
    operator_match = re.search(r"embedding\s+(vector_[a-z_]+_ops)", definition, re.IGNORECASE)
    if operator_match:
        configuration["operator_class"] = operator_match.group(1)
    configuration["partial_predicate"] = "embedding IS NOT NULL" if "WHERE" in definition.upper() else None
    return configuration


def explain_index_usage(
    cursor: Cursor,
    search_sql: str,
    query: str,
    dimension: int,
    top_k: int,
    query_embedder: QueryEmbedder,
) -> dict[str, Any]:
    vector = query_embedder(query, dimension)
    cursor.execute(
        "EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) " + search_sql,
        {"embedding": vector, "limit": top_k},
    )
    row = cursor.fetchone()
    raw_plan = _row_value(row, "QUERY PLAN", 0)
    if isinstance(raw_plan, str):
        raw_plan = json.loads(raw_plan)
    root = raw_plan[0] if isinstance(raw_plan, list) else raw_plan
    plan = root.get("Plan", root) if isinstance(root, dict) else {}
    node_types: list[str] = []
    index_names: list[str] = []
    sequential_scan_tables: list[str] = []
    _collect_plan_nodes(plan, node_types, index_names, sequential_scan_tables)
    index_used = INDEX_NAME in index_names
    return {
        "index_used": index_used,
        "hnsw_index_used": index_used,
        "sequential_scan_used": TABLE_NAME in sequential_scan_tables,
        "node_types": node_types,
        "index_names": index_names,
        "sequential_scan_tables": sequential_scan_tables,
        "planning_time_ms": root.get("Planning Time") if isinstance(root, dict) else None,
        "execution_time_ms": root.get("Execution Time") if isinstance(root, dict) else None,
    }


def latency_statistics(values: Sequence[float]) -> dict[str, float]:
    if not values:
        raise ValueError("at least one latency value is required")
    ordered = sorted(float(value) for value in values)
    return {
        "mean": statistics.fmean(ordered),
        "median": statistics.median(ordered),
        "p95": _percentile(ordered, 0.95),
        "min": ordered[0],
        "max": ordered[-1],
    }


def save_results(results: dict[str, Any], output_path: str | Path = DEFAULT_OUTPUT) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def default_connection_factory() -> Any:
    try:
        import psycopg
        from psycopg.rows import dict_row
    except ImportError as error:
        raise RuntimeError("Real benchmarking requires psycopg; install requirements.txt") from error
    url = os.environ.get(
        "DATABASE_URL",
        "postgresql+asyncpg://postgres:postgres@127.0.0.1:55432/incident_news_search",
    ).replace("postgresql+asyncpg://", "postgresql://", 1)
    try:
        return psycopg.connect(url, row_factory=dict_row)
    except psycopg.OperationalError as error:
        raise RuntimeError(
            "PostgreSQL is unavailable. Start the project database and verify DATABASE_URL."
        ) from error


def default_query_embedder(query: str, dimension: int) -> str:
    # Import lazily so schema-only tests do not require database dependencies.
    try:
        from retrieval.embeddings import build_query_text, validate_embedding_dimension
        from retrieval.query_rewrite import rewrite_incident_query
        from retrieval.db_search import _vector_literal
    except ImportError as error:
        raise RuntimeError("Real benchmarking requires the dependencies in requirements.txt") from error

    vector = _benchmark_embedding_client(dimension).embed_text(build_query_text(rewrite_incident_query(query)))
    validate_embedding_dimension(vector, dimension, _benchmark_embedding_client(dimension).model_name)
    return _vector_literal(vector)


def default_embedding_generator(limit: int) -> int:
    try:
        from data.embed_structured_events import embed_structured_events
    except ImportError as error:
        raise RuntimeError("Embedding generation requires the dependencies in requirements.txt") from error

    return embed_structured_events(limit)


@lru_cache(maxsize=None)
def _benchmark_embedding_client(dimension: int):
    try:
        from retrieval.embeddings import build_embedding_client
    except ImportError as error:
        raise RuntimeError("Real benchmarking requires the dependencies in requirements.txt") from error

    return build_embedding_client(
        backend=os.environ.get("EMBEDDING_BACKEND", "auto"),
        model=os.environ.get("EMBEDDING_MODEL", "intfloat/e5-small-v2"),
        dimensions=dimension,
    )


def default_search_sql() -> str:
    try:
        from retrieval.db_search import PGVECTOR_SQL
    except ImportError as error:
        raise RuntimeError("Real benchmarking requires the dependencies in requirements.txt") from error

    return PGVECTOR_SQL


def format_summary(results: dict[str, Any], output_path: str | Path) -> str:
    hardware = results["platform"]
    cpu = hardware["cpu"]
    gpu = hardware["gpu"]
    embedding = results["embedding"]
    index = results["index"]
    search = results["search"]
    explain = results["explain_analyze"]
    generation_time = embedding["total_generation_time_seconds"]
    generation_text = "not run (read-only)" if generation_time is None else f"{generation_time:.3f} s"
    index_build_time = index["build_time_seconds"]
    index_build_text = "not run" if index_build_time is None else f"{index_build_time:.3f} s"
    lines = [
            "Real PostgreSQL/pgvector retrieval benchmark",
            f"  OS: {hardware['system']} {hardware['release']}; Python {hardware['python_version']}",
            f"  CPU: {cpu['model']} ({cpu['logical_cores']} logical, {cpu['physical_cores']} physical cores)",
            f"  RAM: {_format_bytes(hardware['total_ram_bytes'])}",
            f"  GPU: {gpu['model']}" + (f" ({_format_bytes(gpu['vram_bytes'])} VRAM)" if gpu['vram_bytes'] else ""),
            f"  Documents: {embedding['embedded_documents']} embedded / {embedding['total_documents']} total",
            f"  Embedding: {embedding['provider']} {embedding['model']}, {embedding['vector_dimension']} dimensions",
            f"  Database embedding generation: {generation_text}",
            f"  Index: {index['type']} {index['name']}, {_format_bytes(index['size_on_disk_bytes'])}",
            f"  Index rebuild: {index_build_text}",
            f"  Search: top-{search['top_k']}, {search['warmup_searches']} warm-up + {search['measured_searches']} measured",
            _latency_line("Query embedding", search["query_embedding_latency_ms"]),
            _latency_line("Database/index search", search["database_index_search_latency_ms"]),
            _latency_line("End-to-end retrieval", search["end_to_end_retrieval_latency_ms"]),
            f"  EXPLAIN ANALYZE: HNSW index used={explain['hnsw_index_used']}, sequential scan used={explain['sequential_scan_used']}",
    ]
    document_benchmark = results.get("document_embedding_benchmark")
    if document_benchmark:
        lines.extend(
            (
                f"  Document embeddings: {document_benchmark['sample_size']} in "
                f"{document_benchmark['total_generation_time_seconds']:.6f} s, "
                f"{document_benchmark['throughput_documents_per_second']:.2f} docs/s",
                _latency_line("Document embedding", document_benchmark["per_document_latency_ms"]),
                "  Document embedding persistence: none (read-only, in memory only)",
            )
        )
    lines.append(f"  Results: {output_path}")
    return "\n".join(lines)


def _latency_line(label: str, stats: dict[str, float]) -> str:
    return (
        f"  {label} (ms): mean={stats['mean']:.3f}, median={stats['median']:.3f}, "
        f"p95={stats['p95']:.3f}, min={stats['min']:.3f}, max={stats['max']:.3f}"
    )


def _collect_plan_nodes(
    node: dict[str, Any],
    node_types: list[str],
    index_names: list[str],
    sequential_scan_tables: list[str],
) -> None:
    node_type = str(node.get("Node Type") or "")
    if node_type:
        node_types.append(node_type)
    if node.get("Index Name"):
        index_names.append(str(node["Index Name"]))
    if node_type == "Seq Scan" and node.get("Relation Name"):
        sequential_scan_tables.append(str(node["Relation Name"]))
    for child in node.get("Plans", []):
        _collect_plan_nodes(child, node_types, index_names, sequential_scan_tables)


def _row_value(row: Any, key: str, position: int) -> Any:
    if isinstance(row, dict):
        return row.get(key)
    return row[position]


def _numeric_value(value: str) -> int | float | str:
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def _configured_dimension() -> int:
    return int(os.environ.get("EMBEDDING_DIM", "384"))


def _percentile(ordered: Sequence[float], fraction: float) -> float:
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _format_bytes(value: int | None) -> str:
    if value is None:
        return "Unknown"
    size = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024.0 or unit == "TiB":
            return f"{size:.2f} {unit}"
        size /= 1024.0
    raise AssertionError("unreachable")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark the real PostgreSQL/pgvector retrieval pipeline.")
    parser.add_argument("--validation-set", type=Path, default=DEFAULT_VALIDATION_SET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--searches", type=int, default=DEFAULT_SEARCHES)
    parser.add_argument("--warmups", type=int, default=DEFAULT_WARMUPS)
    parser.add_argument(
        "--generate-embeddings",
        action="store_true",
        help="Generate embeddings only for currently unembedded structured events (modifies the database).",
    )
    parser.add_argument("--embedding-limit", type=int, default=DEFAULT_EMBEDDING_LIMIT)
    parser.add_argument(
        "--rebuild-index",
        action="store_true",
        help=f"Run REINDEX on {INDEX_NAME} before benchmarking (modifies the database).",
    )
    parser.add_argument(
        "--benchmark-document-embeddings",
        action="store_true",
        help="Benchmark production document embeddings in memory without database writes.",
    )
    parser.add_argument(
        "--embedding-sample-size",
        type=int,
        default=DEFAULT_EMBEDDING_SAMPLE_SIZE,
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    try:
        results = run_real_benchmark(
            validation_path=args.validation_set,
            output_path=args.output,
            top_k=args.top_k,
            searches=args.searches,
            warmups=args.warmups,
            generate_embeddings=args.generate_embeddings,
            embedding_limit=args.embedding_limit,
            rebuild_index=args.rebuild_index,
            benchmark_document_embeddings=args.benchmark_document_embeddings,
            embedding_sample_size=args.embedding_sample_size,
        )
    except RuntimeError as error:
        raise SystemExit(f"Real benchmark failed: {error}") from error
    print(format_summary(results, args.output))


if __name__ == "__main__":
    main()
