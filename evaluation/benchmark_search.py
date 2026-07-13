from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from retrieval.db_search import DbNewsHit, DbNewsSearchService, SearchMode


DEFAULT_VALIDATION_SET = Path("evaluation/data/validation_set.json")
DEFAULT_OUTPUT = Path("evaluation/benchmark_results.json")


def benchmark(validation_path: Path, top_k: int, max_queries: int) -> dict:
    payload = json.loads(validation_path.read_text(encoding="utf-8"))
    examples = payload["examples"][:max_queries]
    service = DbNewsSearchService()
    modes: tuple[SearchMode, ...] = ("bm25", "dense", "pgvector", "hybrid")
    result = {"queries": len(examples), "top_k": top_k, "modes": {}}

    cached_hits: dict[str, dict[str, list[DbNewsHit]]] = {example["id"]: {} for example in examples}
    cached_latency: dict[str, dict[str, float]] = {example["id"]: {} for example in examples}

    for example in examples:
        for mode in ("bm25", "dense", "pgvector"):
            started = time.perf_counter()
            cached_hits[example["id"]][mode] = service.search(example["query"]["message"], mode=mode, top_k=top_k)
            cached_latency[example["id"]][mode] = (time.perf_counter() - started) * 1000
        started = time.perf_counter()
        cached_hits[example["id"]]["hybrid"] = _fuse_hits(
            cached_hits[example["id"]]["bm25"],
            cached_hits[example["id"]]["dense"],
            cached_hits[example["id"]]["pgvector"],
            top_k,
        )
        cached_latency[example["id"]]["hybrid"] = (
            cached_latency[example["id"]]["bm25"]
            + cached_latency[example["id"]]["dense"]
            + cached_latency[example["id"]]["pgvector"]
            + (time.perf_counter() - started) * 1000
        )

    for mode in modes:
        latencies = []
        total_hits = 0
        for example in examples:
            hits = cached_hits[example["id"]][mode]
            latencies.append(cached_latency[example["id"]][mode])
            total_hits += len(hits)
        result["modes"][mode] = {
            "mean_latency_ms": statistics.fmean(latencies) if latencies else 0.0,
            "p50_latency_ms": _percentile(latencies, 50),
            "p95_latency_ms": _percentile(latencies, 95),
            "avg_results": total_hits / len(examples) if examples else 0.0,
        }
    return result


def _percentile(values: list[float], percentile: int) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((percentile / 100) * (len(ordered) - 1))))
    return ordered[index]


def _fuse_hits(bm25: list[DbNewsHit], dense: list[DbNewsHit], pgvector: list[DbNewsHit], top_k: int) -> list[DbNewsHit]:
    by_id: dict[str, DbNewsHit] = {}
    scores: dict[str, float] = {}
    for hits in (bm25, dense, pgvector):
        for hit in hits:
            by_id.setdefault(hit.id, hit)
            scores[hit.id] = scores.get(hit.id, 0.0) + 1.0 / (60 + hit.rank)
    ranked_ids = sorted(scores, key=lambda item_id: scores[item_id], reverse=True)[:top_k]
    return [
        DbNewsHit(
            **{
                **by_id[item_id].__dict__,
                "score": scores[item_id],
                "rank": rank,
                "method": "hybrid",
            }
        )
        for rank, item_id in enumerate(ranked_ids, start=1)
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark search latency on validation queries.")
    parser.add_argument("--validation-set", type=Path, default=DEFAULT_VALIDATION_SET)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-queries", type=int, default=60)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = benchmark(args.validation_set, args.top_k, args.max_queries)
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"output: {args.output}")


if __name__ == "__main__":
    main()
