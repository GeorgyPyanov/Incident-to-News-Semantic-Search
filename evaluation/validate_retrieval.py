from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from retrieval.db_search import DbNewsHit, DbNewsSearchService, SearchMode


DEFAULT_VALIDATION_SET = Path("evaluation/data/validation_set.json")
DEFAULT_RESULTS = Path("evaluation/validation_results.json")
MODES: tuple[SearchMode, ...] = ("bm25", "dense", "pgvector", "hybrid")


def evaluate(validation_path: Path, top_k: int, workers: int = 1) -> dict:
    if workers < 1:
        raise ValueError("workers must be positive")
    payload = json.loads(validation_path.read_text(encoding="utf-8"))
    examples = payload["examples"]
    evaluated_examples = _evaluate_examples(examples, top_k=top_k, workers=workers)
    results = {
        "examples": len(examples),
        "top_k": top_k,
        "workers": workers,
        "runtime_config": _runtime_config(),
        "modes": {},
    }

    for mode in MODES:
        hits = 0
        reciprocal_ranks: list[float] = []
        recalls: list[float] = []
        ndcgs: list[float] = []
        latencies_ms: list[float] = []
        negative_hits = 0
        per_dataset: dict[str, dict[str, int]] = {}
        misses = []
        for example in evaluated_examples:
            dataset = example["dataset"]
            per_dataset.setdefault(dataset, {"examples": 0, "hits": 0})
            per_dataset[dataset]["examples"] += 1

            relevant_ids = example["relevant_ids"]
            negative_ids = example["negative_ids"]
            hits_for_query = example["hits"][mode]
            latencies_ms.append(example["latency_ms"][mode])
            found_ids = [hit.id for hit in hits_for_query]
            found_set = set(found_ids)
            matched_ranks = [rank for rank, item_id in enumerate(found_ids, start=1) if item_id in relevant_ids]
            is_hit = bool(matched_ranks)
            if is_hit:
                hits += 1
                per_dataset[dataset]["hits"] += 1
                reciprocal_ranks.append(1.0 / min(matched_ranks))
            elif len(misses) < 10:
                reciprocal_ranks.append(0.0)
                misses.append(
                    {
                        "example_id": example["id"],
                        "dataset": dataset,
                        "relevant_news_ids": sorted(relevant_ids),
                    }
                )
            else:
                reciprocal_ranks.append(0.0)

            recall = len(found_set & relevant_ids) / len(relevant_ids) if relevant_ids else 0.0
            recalls.append(recall)
            ndcgs.append(_ndcg(found_ids, relevant_ids, top_k))
            if found_set & negative_ids:
                negative_hits += 1

        results["modes"][mode] = {
            "hit_at_k": hits / len(examples) if examples else 0.0,
            "mrr_at_k": statistics.fmean(reciprocal_ranks) if reciprocal_ranks else 0.0,
            "recall_at_k": statistics.fmean(recalls) if recalls else 0.0,
            "ndcg_at_k": statistics.fmean(ndcgs) if ndcgs else 0.0,
            "hits": hits,
            "negative_hit_at_k": negative_hits / len(examples) if examples else 0.0,
            "latency_ms": {
                "mean": statistics.fmean(latencies_ms) if latencies_ms else 0.0,
                "p50": _percentile(latencies_ms, 50),
                "p95": _percentile(latencies_ms, 95),
            },
            "per_dataset": {
                dataset: {
                    **values,
                    "hit_at_k": values["hits"] / values["examples"] if values["examples"] else 0.0,
                }
                for dataset, values in per_dataset.items()
            },
            "sample_misses": misses,
        }
    return results


def _evaluate_examples(examples: list[dict], *, top_k: int, workers: int) -> list[dict]:
    state = threading.local()
    services: list[DbNewsSearchService] = []
    services_lock = threading.Lock()

    def service_for_thread() -> DbNewsSearchService:
        service = getattr(state, "service", None)
        if service is None:
            service = DbNewsSearchService()
            state.service = service
            with services_lock:
                services.append(service)
        return service

    def evaluate_example(example: dict) -> dict:
        service = service_for_thread()
        hits: dict[str, list[DbNewsHit]] = {}
        latency_ms: dict[str, float] = {}
        for mode in MODES:
            started = time.perf_counter()
            hits[mode] = service.search(example["query"]["message"], mode=mode, top_k=top_k)
            latency_ms[mode] = (time.perf_counter() - started) * 1000
        return {
            "id": example["id"],
            "dataset": example["query"]["dataset"],
            "relevant_ids": {item["news_id"] for item in example["relevant_news"]},
            "negative_ids": {item["news_id"] for item in example.get("negative_news", [])},
            "hits": hits,
            "latency_ms": latency_ms,
        }

    try:
        if workers == 1:
            return [evaluate_example(example) for example in examples]
        with ThreadPoolExecutor(max_workers=workers) as executor:
            return list(executor.map(evaluate_example, examples))
    finally:
        for service in services:
            service.close()


def _runtime_config() -> dict[str, object]:
    return {
        "embedding_backend": os.getenv("EMBEDDING_BACKEND", "auto"),
        "embedding_model": os.getenv("EMBEDDING_MODEL", "intfloat/e5-small-v2"),
        "fusion_mode": os.getenv("RETRIEVAL_FUSION_MODE", "rrf"),
        "deepseek_rerank_enabled": str(os.getenv("DEEPSEEK_RERANK_ENABLED", "")).strip().lower()
        in {"1", "true", "yes", "on"},
        "deepseek_model": os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
        "deepseek_rerank_top_n": int(os.getenv("DEEPSEEK_RERANK_TOP_N", "12")),
    }


def _ndcg(found_ids: list[str], relevant_ids: set[str], top_k: int) -> float:
    dcg = 0.0
    for rank, item_id in enumerate(found_ids[:top_k], start=1):
        if item_id in relevant_ids:
            dcg += 1.0 / math.log2(rank + 1)
    ideal_relevant = min(len(relevant_ids), top_k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_relevant + 1))
    return dcg / idcg if idcg else 0.0


def _percentile(values: list[float], percentile: int) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil((percentile / 100) * len(ordered)) - 1))
    return ordered[index]
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate DB search against the validation set.")
    parser.add_argument("--validation-set", type=Path, default=DEFAULT_VALIDATION_SET)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--workers", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = evaluate(args.validation_set, args.top_k, workers=args.workers)
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results["modes"], ensure_ascii=False, indent=2))
    print(f"output: {args.output}")


if __name__ == "__main__":
    main()
