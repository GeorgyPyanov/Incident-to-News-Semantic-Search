"""Benchmark helpers for dense retrieval and HNSW verification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence


@dataclass(slots=True)
class BenchmarkResult:
    label: str
    latency_ms: float
    top_k_ids: list[int]
    plan: str | None = None


def explain_plan(cursor: Any, sql: str, params: dict[str, Any]) -> str:
    # EXPLAIN ANALYZE is the direct way to verify whether HNSW was used.
    cursor.execute("EXPLAIN (ANALYZE, BUFFERS, FORMAT TEXT) " + sql, params)
    rows = cursor.fetchall()
    return "\n".join(row[0] for row in rows)


def run_dense_benchmark(
    cursor: Any,
    sql: str,
    params: dict[str, Any],
    label: str,
    *,
    explain: bool = False,
) -> BenchmarkResult:
    # Measure wall-clock time only around the query execution.
    cursor.execute("SELECT clock_timestamp()")
    start = cursor.fetchone()[0]
    cursor.execute(sql, params)
    rows = cursor.fetchall()
    cursor.execute("SELECT clock_timestamp()")
    end = cursor.fetchone()[0]
    latency_ms = (end - start).total_seconds() * 1000.0
    top_k_ids = [row[0] for row in rows]
    plan = explain_plan(cursor, sql, params) if explain else None
    return BenchmarkResult(label=label, latency_ms=latency_ms, top_k_ids=top_k_ids, plan=plan)


def compare_dense_strategies(
    indexed_result: BenchmarkResult,
    seqscan_result: BenchmarkResult,
    relevant_ids: Iterable[int] | None = None,
) -> dict[str, Any]:
    relevant = set(relevant_ids or [])

    def recall(ids: Sequence[int]) -> float | None:
        # Recall is optional because we may not have a labeled relevance set.
        if not relevant:
            return None
        return len(relevant.intersection(ids)) / len(relevant)

    return {
        "indexed_latency_ms": indexed_result.latency_ms,
        "seqscan_latency_ms": seqscan_result.latency_ms,
        "latency_delta_ms": seqscan_result.latency_ms - indexed_result.latency_ms,
        "indexed_recall": recall(indexed_result.top_k_ids),
        "seqscan_recall": recall(seqscan_result.top_k_ids),
        "indexed_ids": indexed_result.top_k_ids,
        "seqscan_ids": seqscan_result.top_k_ids,
        "indexed_plan": indexed_result.plan,
        "seqscan_plan": seqscan_result.plan,
    }

