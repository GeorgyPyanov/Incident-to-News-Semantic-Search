from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path

from retrieval.db_search import DbNewsHit, DbNewsSearchService, SearchMode


DEFAULT_VALIDATION_SET = Path("evaluation/data/validation_blind.json")
DEFAULT_QRELS = Path("evaluation/data/qrels.jsonl")
DEFAULT_OUTPUT = Path("evaluation/qrels_validation_results.json")


def evaluate(validation_path: Path, qrels_path: Path, top_k: int) -> dict:
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    examples = validation["examples"]
    qrels = _load_qrels(qrels_path)
    service = DbNewsSearchService()
    modes: tuple[SearchMode, ...] = ("bm25", "dense", "pgvector", "hybrid")
    raw_hits: dict[str, dict[str, list[DbNewsHit]]] = {example["id"]: {} for example in examples}
    raw_latency: dict[str, dict[str, float]] = {example["id"]: {} for example in examples}

    for example in examples:
        query_id = example["id"]
        for mode in ("bm25", "dense", "pgvector", "hybrid"):
            started = time.perf_counter()
            raw_hits[query_id][mode] = service.search(example["query"]["message"], mode=mode, top_k=top_k)
            raw_latency[query_id][mode] = (time.perf_counter() - started) * 1000

    results = {"validation_set": str(validation_path), "qrels": str(qrels_path), "queries": len(examples), "top_k": top_k, "modes": {}}
    for mode in modes:
        ndcgs: list[float] = []
        mrrs: list[float] = []
        recalls: list[float] = []
        precision_values: list[float] = []
        hard_negative_queries = 0
        latencies: list[float] = []
        per_dataset: dict[str, dict[str, float]] = {}

        for example in examples:
            query_id = example["id"]
            labels = qrels.get(query_id, {})
            relevant_docs = {doc_id for doc_id, rel in labels.items() if rel > 0}
            high_relevance_docs = {doc_id for doc_id, rel in labels.items() if rel >= 3}
            hard_negative_docs = {doc_id for doc_id, rel in labels.items() if rel == 1}
            found_ids = [hit.id for hit in raw_hits[query_id][mode]]
            found_set = set(found_ids)
            latencies.append(raw_latency[query_id][mode])

            ndcg = _ndcg(found_ids, labels, top_k)
            mrr = _mrr(found_ids, high_relevance_docs)
            recall = len(found_set & relevant_docs) / len(relevant_docs) if relevant_docs else 0.0
            precision = len(found_set & relevant_docs) / min(top_k, len(found_ids)) if found_ids else 0.0
            hard_negative_hit = bool(found_set & hard_negative_docs)
            if hard_negative_hit:
                hard_negative_queries += 1

            ndcgs.append(ndcg)
            mrrs.append(mrr)
            recalls.append(recall)
            precision_values.append(precision)

            dataset = example["query"]["dataset"]
            bucket = per_dataset.setdefault(dataset, {"queries": 0, "ndcg_sum": 0.0, "mrr_sum": 0.0})
            bucket["queries"] += 1
            bucket["ndcg_sum"] += ndcg
            bucket["mrr_sum"] += mrr

        results["modes"][mode] = {
            "ndcg_at_k": statistics.fmean(ndcgs) if ndcgs else 0.0,
            "mrr_at_k": statistics.fmean(mrrs) if mrrs else 0.0,
            "recall_at_k": statistics.fmean(recalls) if recalls else 0.0,
            "precision_at_k": statistics.fmean(precision_values) if precision_values else 0.0,
            "hard_negative_hit_at_k": hard_negative_queries / len(examples) if examples else 0.0,
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = evaluate(args.validation_set, args.qrels, args.top_k)
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results["modes"], ensure_ascii=False, indent=2))
    print(f"output: {args.output}")


if __name__ == "__main__":
    main()
