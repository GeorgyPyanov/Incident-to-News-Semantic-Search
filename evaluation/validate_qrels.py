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


DEFAULT_VALIDATION_SET = Path("evaluation/data/validation_blind.json")
DEFAULT_QRELS = Path("evaluation/data/qrels.jsonl")
DEFAULT_OUTPUT = Path("evaluation/qrels_validation_results.json")
MODES: tuple[SearchMode, ...] = ("bm25", "dense", "pgvector", "hybrid")


def evaluate(validation_path: Path, qrels_path: Path, top_k: int, workers: int = 1) -> dict:
    if workers < 1:
        raise ValueError("workers must be positive")
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    examples = validation["examples"]
    qrels = _load_qrels(qrels_path)
    evaluated_examples = _evaluate_examples(examples, qrels=qrels, top_k=top_k, workers=workers)

    results = {
        "validation_set": str(validation_path),
        "qrels": str(qrels_path),
        "queries": len(examples),
        "top_k": top_k,
        "workers": workers,
        "runtime_config": _runtime_config(),
        "modes": {},
    }
    for mode in MODES:
        ndcgs: list[float] = []
        mrrs: list[float] = []
        recalls: list[float] = []
        precision_values: list[float] = []
        latencies: list[float] = []
        per_dataset: dict[str, dict[str, float]] = {}

        for example in evaluated_examples:
            labels = example["labels"]
            relevant_docs = {doc_id for doc_id, rel in labels.items() if rel > 0}
            high_relevance_docs = {doc_id for doc_id, rel in labels.items() if rel >= 3}
            found_ids = [hit.id for hit in example["hits"][mode]]
            found_set = set(found_ids)
            latencies.append(example["latency_ms"][mode])

            ndcg = _ndcg(found_ids, labels, top_k)
            mrr = _mrr(found_ids, high_relevance_docs)
            recall = len(found_set & relevant_docs) / len(relevant_docs) if relevant_docs else 0.0
            precision = len(found_set & relevant_docs) / min(top_k, len(found_ids)) if found_ids else 0.0

            ndcgs.append(ndcg)
            mrrs.append(mrr)
            recalls.append(recall)
            precision_values.append(precision)

            dataset = example["dataset"]
            bucket = per_dataset.setdefault(dataset, {"queries": 0, "ndcg_sum": 0.0, "mrr_sum": 0.0})
            bucket["queries"] += 1
            bucket["ndcg_sum"] += ndcg
            bucket["mrr_sum"] += mrr

        results["modes"][mode] = {
            "ndcg_at_k": statistics.fmean(ndcgs) if ndcgs else 0.0,
            "mrr_at_k": statistics.fmean(mrrs) if mrrs else 0.0,
            "recall_at_k": statistics.fmean(recalls) if recalls else 0.0,
            "precision_at_k": statistics.fmean(precision_values) if precision_values else 0.0,
            "latency_ms": {
                "mean": statistics.fmean(latencies) if latencies else 0.0,
                "p50": _percentile(latencies, 50),
                "p95": _percentile(latencies, 95),
            },
            "per_dataset": {
                dataset: {
                    "queries": int(values["queries"]),
                    "ndcg_at_k": values["ndcg_sum"] / values["queries"] if values["queries"] else 0.0,
                    "mrr_at_k": values["mrr_sum"] / values["queries"] if values["queries"] else 0.0,
                }
                for dataset, values in per_dataset.items()
            },
        }
    return results


def _evaluate_examples(examples: list[dict], *, qrels: dict[str, dict[str, int]], top_k: int, workers: int) -> list[dict]:
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
            "labels": qrels.get(example["id"], {}),
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


def _load_qrels(path: Path) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        qrels.setdefault(row["query_id"], {})[row["doc_id"]] = int(row["relevance"])
    return qrels


def _ndcg(found_ids: list[str], labels: dict[str, int], top_k: int) -> float:
    dcg = 0.0
    for rank, doc_id in enumerate(found_ids[:top_k], start=1):
        relevance = labels.get(doc_id, 0)
        gain = (2**relevance - 1) / math.log2(rank + 1)
        dcg += gain
    ideal = sorted(labels.values(), reverse=True)[:top_k]
    idcg = sum((2**relevance - 1) / math.log2(rank + 1) for rank, relevance in enumerate(ideal, start=1))
    return dcg / idcg if idcg else 0.0


def _mrr(found_ids: list[str], relevant_ids: set[str]) -> float:
    for rank, doc_id in enumerate(found_ids, start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def _percentile(values: list[float], percentile: int) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil((percentile / 100) * len(ordered)) - 1))
    return ordered[index]
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate retrieval with graded qrels.")
    parser.add_argument("--validation-set", type=Path, default=DEFAULT_VALIDATION_SET)
    parser.add_argument("--qrels", type=Path, default=DEFAULT_QRELS)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = evaluate(args.validation_set, args.qrels, args.top_k, workers=args.workers)
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results["modes"], ensure_ascii=False, indent=2))
    print(f"output: {args.output}")


if __name__ == "__main__":
    main()
