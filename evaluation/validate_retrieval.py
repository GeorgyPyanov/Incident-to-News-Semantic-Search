from __future__ import annotations

import argparse
import json
from pathlib import Path

from retrieval.db_search import DbNewsSearchService, SearchMode


DEFAULT_VALIDATION_SET = Path("evaluation/data/validation_set.json")
DEFAULT_RESULTS = Path("evaluation/validation_results.json")


def evaluate(validation_path: Path, top_k: int) -> dict:
    payload = json.loads(validation_path.read_text(encoding="utf-8"))
    examples = payload["examples"]
    service = DbNewsSearchService()
    modes: tuple[SearchMode, ...] = ("bm25", "dense", "hybrid")
    results = {"examples": len(examples), "top_k": top_k, "modes": {}}

    for mode in modes:
        hits = 0
        per_dataset: dict[str, dict[str, int]] = {}
        misses = []
        for example in examples:
            dataset = example["query"]["dataset"]
            per_dataset.setdefault(dataset, {"examples": 0, "hits": 0})
            per_dataset[dataset]["examples"] += 1

            relevant_ids = {item["news_id"] for item in example["relevant_news"]}
            found_ids = {hit.id for hit in service.search(example["query"]["message"], mode=mode, top_k=top_k)}
            is_hit = bool(relevant_ids & found_ids)
            if is_hit:
                hits += 1
                per_dataset[dataset]["hits"] += 1
            elif len(misses) < 10:
                misses.append(
                    {
                        "example_id": example["id"],
                        "dataset": dataset,
                        "relevant_news_ids": sorted(relevant_ids),
                    }
                )

        results["modes"][mode] = {
            "hit_at_k": hits / len(examples) if examples else 0.0,
            "hits": hits,
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
