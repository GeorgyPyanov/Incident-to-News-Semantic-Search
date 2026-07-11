from __future__ import annotations

import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable

from evaluation.dataset import DEFAULT_DATASET_PATH, EvaluationDataset, load_evaluation_dataset
from evaluation.metrics import calculate_metrics
from evaluation.results import format_comparison_table, save_results_csv, save_results_json
from evaluation.retrievers import EvaluationRetriever, default_retrieval_approaches


DEFAULT_K_VALUES = (1, 3, 5)
DEFAULT_OUTPUT_DIR = Path(__file__).parent


def run_evaluation(
    dataset: EvaluationDataset | None = None,
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    k_values: tuple[int, ...] = DEFAULT_K_VALUES,
    approaches: tuple[EvaluationRetriever, ...] | None = None,
    timestamp: str | None = None,
    clock: Callable[[], float] = time.perf_counter,
    save: bool = True,
) -> list[dict[str, object]]:
    evaluation_dataset = dataset or load_evaluation_dataset(dataset_path)
    selected_approaches = approaches or default_retrieval_approaches()
    evaluation_timestamp = timestamp or datetime.now(UTC).isoformat()
    max_k = max(k_values)

    results = [
        _evaluate_approach(approach, evaluation_dataset, k_values, max_k, evaluation_timestamp, clock)
        for approach in selected_approaches
    ]

    if save:
        output_path = Path(output_dir)
        save_results_json(results, output_path / "results.json")
        save_results_csv(results, output_path / "results.csv")

    return results


def main() -> None:
    results = run_evaluation()
    print(format_comparison_table(results))


def _evaluate_approach(
    approach: EvaluationRetriever,
    dataset: EvaluationDataset,
    k_values: tuple[int, ...],
    max_k: int,
    timestamp: str,
    clock: Callable[[], float],
) -> dict[str, object]:
    ranked_ids_by_query: list[list[str]] = []
    relevant_ids_by_query: list[frozenset[str]] = []
    per_query_results: list[dict[str, object]] = []
    total_seconds = 0.0

    for query in dataset.queries:
        started_at = clock()
        retrieved = approach.search(query.incident_log, query.candidate_articles, top_k=max_k)
        elapsed_seconds = clock() - started_at
        total_seconds += max(elapsed_seconds, 0.0)

        ranked_ids = [result.article.id for result in retrieved]
        ranked_ids_by_query.append(ranked_ids)
        relevant_ids_by_query.append(query.relevant_article_ids)
        per_query_results.append(
            {
                "query_id": query.id,
                "ranked_article_ids": ranked_ids,
                "relevant_article_ids": sorted(query.relevant_article_ids),
                "scores": {result.article.id: result.score for result in retrieved},
                "execution_time_ms": elapsed_seconds * 1000,
            }
        )

    metrics = calculate_metrics(ranked_ids_by_query, relevant_ids_by_query, k_values)
    query_count = len(dataset.queries)
    average_execution_time_ms = (total_seconds / query_count * 1000) if query_count else 0.0

    return {
        "approach_name": approach.name,
        "metrics": metrics,
        "selected_k_values": list(k_values),
        "average_execution_time_ms": average_execution_time_ms,
        "num_queries": query_count,
        "evaluation_timestamp": timestamp,
        "configuration": approach.config,
        "per_query_results": per_query_results,
    }


if __name__ == "__main__":
    main()

