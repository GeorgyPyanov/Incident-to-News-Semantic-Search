from __future__ import annotations

import math
from collections.abc import Iterable, Sequence


def precision_at_k(ranked_ids: Sequence[str], relevant_ids: Iterable[str], k: int) -> float:
    _validate_k(k)
    relevant = set(relevant_ids)
    if not relevant:
        return 0.0
    return _hits(ranked_ids, relevant, k) / k


def recall_at_k(ranked_ids: Sequence[str], relevant_ids: Iterable[str], k: int) -> float:
    _validate_k(k)
    relevant = set(relevant_ids)
    if not relevant:
        return 0.0
    return _hits(ranked_ids, relevant, k) / len(relevant)


def reciprocal_rank(ranked_ids: Sequence[str], relevant_ids: Iterable[str]) -> float:
    relevant = set(relevant_ids)
    if not relevant:
        return 0.0

    for index, article_id in enumerate(ranked_ids, start=1):
        if article_id in relevant:
            return 1.0 / index
    return 0.0


def average_precision(ranked_ids: Sequence[str], relevant_ids: Iterable[str]) -> float:
    relevant = set(relevant_ids)
    if not relevant:
        return 0.0

    score = 0.0
    hits = 0
    for index, article_id in enumerate(ranked_ids, start=1):
        if article_id in relevant:
            hits += 1
            score += hits / index
    return score / len(relevant)


def ndcg_at_k(ranked_ids: Sequence[str], relevant_ids: Iterable[str], k: int) -> float:
    _validate_k(k)
    relevant = set(relevant_ids)
    if not relevant:
        return 0.0

    dcg = 0.0
    for index, article_id in enumerate(ranked_ids[:k], start=1):
        if article_id in relevant:
            dcg += 1.0 / math.log2(index + 1)

    ideal_hits = min(len(relevant), k)
    ideal_dcg = sum(1.0 / math.log2(index + 1) for index in range(1, ideal_hits + 1))
    return dcg / ideal_dcg if ideal_dcg else 0.0


def calculate_metrics(
    ranked_ids_by_query: Sequence[Sequence[str]],
    relevant_ids_by_query: Sequence[Iterable[str]],
    k_values: Sequence[int],
) -> dict[str, float]:
    if len(ranked_ids_by_query) != len(relevant_ids_by_query):
        raise ValueError("ranked and relevant query counts must match")

    query_count = len(ranked_ids_by_query)
    if query_count == 0:
        return _empty_metrics(k_values)

    metrics: dict[str, float] = {}
    for k in k_values:
        _validate_k(k)
        metrics[f"precision@{k}"] = _mean(
            precision_at_k(ranked, relevant, k)
            for ranked, relevant in zip(ranked_ids_by_query, relevant_ids_by_query)
        )
        metrics[f"recall@{k}"] = _mean(
            recall_at_k(ranked, relevant, k)
            for ranked, relevant in zip(ranked_ids_by_query, relevant_ids_by_query)
        )
        metrics[f"ndcg@{k}"] = _mean(
            ndcg_at_k(ranked, relevant, k)
            for ranked, relevant in zip(ranked_ids_by_query, relevant_ids_by_query)
        )

    metrics["mrr"] = _mean(
        reciprocal_rank(ranked, relevant)
        for ranked, relevant in zip(ranked_ids_by_query, relevant_ids_by_query)
    )
    metrics["map"] = _mean(
        average_precision(ranked, relevant)
        for ranked, relevant in zip(ranked_ids_by_query, relevant_ids_by_query)
    )
    return metrics


def _hits(ranked_ids: Sequence[str], relevant: set[str], k: int) -> int:
    return sum(1 for article_id in ranked_ids[:k] if article_id in relevant)


def _mean(values: Iterable[float]) -> float:
    materialized = tuple(values)
    return sum(materialized) / len(materialized) if materialized else 0.0


def _empty_metrics(k_values: Sequence[int]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for k in k_values:
        metrics[f"precision@{k}"] = 0.0
        metrics[f"recall@{k}"] = 0.0
        metrics[f"ndcg@{k}"] = 0.0
    metrics["mrr"] = 0.0
    metrics["map"] = 0.0
    return metrics


def _validate_k(k: int) -> None:
    if k <= 0:
        raise ValueError("k must be positive")

