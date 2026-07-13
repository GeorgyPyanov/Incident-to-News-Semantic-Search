from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path

from retrieval.db_search import DbNewsHit, DbNewsSearchService, SearchMode


DEFAULT_VALIDATION_SET = Path("evaluation/data/validation_set.json")
DEFAULT_RESULTS = Path("evaluation/validation_results.json")


def evaluate(validation_path: Path, top_k: int) -> dict:
    payload = json.loads(validation_path.read_text(encoding="utf-8"))
    examples = payload["examples"]
    service = DbNewsSearchService()
    modes: tuple[SearchMode, ...] = ("bm25", "dense", "pgvector", "hybrid")
    raw_hits: dict[str, dict[str, list[DbNewsHit]]] = {
        example["id"]: {} for example in examples
    }
    raw_latency: dict[str, dict[str, float]] = {
        example["id"]: {} for example in examples
    }

    for example in examples:
        for mode in ("bm25", "dense", "pgvector"):
            started = time.perf_counter()
            raw_hits[example["id"]][mode] = service.search(example["query"]["message"], mode=mode, top_k=top_k)
            raw_latency[example["id"]][mode] = (time.perf_counter() - started) * 1000
        started = time.perf_counter()
        raw_hits[example["id"]]["hybrid"] = _fuse_hits(
            raw_hits[example["id"]]["bm25"],
            raw_hits[example["id"]]["dense"],
            raw_hits[example["id"]]["pgvector"],
            top_k,
        )
        raw_latency[example["id"]]["hybrid"] = (
            raw_latency[example["id"]]["bm25"]
            + raw_latency[example["id"]]["dense"]
            + raw_latency[example["id"]]["pgvector"]
            + (time.perf_counter() - started) * 1000
        )
    results = {"examples": len(examples), "top_k": top_k, "modes": {}}

    for mode in modes:
        hits = 0
        reciprocal_ranks: list[float] = []
        recalls: list[float] = []
        ndcgs: list[float] = []
        latencies_ms: list[float] = []
        negative_hits = 0
        per_dataset: dict[str, dict[str, int]] = {}
        misses = []
        for example in examples:
            dataset = example["query"]["dataset"]
            per_dataset.setdefault(dataset, {"examples": 0, "hits": 0})
            per_dataset[dataset]["examples"] += 1

            relevant_ids = {item["news_id"] for item in example["relevant_news"]}
            negative_ids = {item["news_id"] for item in example.get("negative_news", [])}
            hits_for_query = raw_hits[example["id"]][mode]
            latencies_ms.append(raw_latency[example["id"]][mode])
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
    parser = argparse.ArgumentParser(description="Evaluate DB search against the validation set.")
    parser.add_argument("--validation-set", type=Path, default=DEFAULT_VALIDATION_SET)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output", type=Path, default=DEFAULT_RESULTS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = evaluate(args.validation_set, args.top_k)
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results["modes"], ensure_ascii=False, indent=2))
    print(f"output: {args.output}")


if __name__ == "__main__":
    main()
