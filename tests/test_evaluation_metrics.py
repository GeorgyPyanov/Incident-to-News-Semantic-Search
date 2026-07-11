from __future__ import annotations

import math
import unittest

from evaluation.metrics import (
    average_precision,
    calculate_metrics,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


class EvaluationMetricsTests(unittest.TestCase):
    def test_calculates_each_metric_on_known_example(self) -> None:
        ranked_ids = ["a", "b", "c", "d"]
        relevant_ids = {"b", "d"}

        self.assertEqual(0.0, precision_at_k(ranked_ids, relevant_ids, 1))
        self.assertAlmostEqual(1 / 3, precision_at_k(ranked_ids, relevant_ids, 3))
        self.assertAlmostEqual(1 / 2, recall_at_k(ranked_ids, relevant_ids, 3))
        self.assertAlmostEqual(1 / 2, reciprocal_rank(ranked_ids, relevant_ids))
        self.assertAlmostEqual(1 / 2, average_precision(ranked_ids, relevant_ids))

        expected_ndcg = (1 / math.log2(3)) / (1 + (1 / math.log2(3)))
        self.assertAlmostEqual(expected_ndcg, ndcg_at_k(ranked_ids, relevant_ids, 3))

    def test_handles_queries_with_no_relevant_documents(self) -> None:
        ranked_ids = ["a", "b"]
        relevant_ids: set[str] = set()

        self.assertEqual(0.0, precision_at_k(ranked_ids, relevant_ids, 1))
        self.assertEqual(0.0, recall_at_k(ranked_ids, relevant_ids, 1))
        self.assertEqual(0.0, reciprocal_rank(ranked_ids, relevant_ids))
        self.assertEqual(0.0, average_precision(ranked_ids, relevant_ids))
        self.assertEqual(0.0, ndcg_at_k(ranked_ids, relevant_ids, 3))

        metrics = calculate_metrics([ranked_ids], [relevant_ids], (1, 3))
        self.assertTrue(all(value == 0.0 for value in metrics.values()))


if __name__ == "__main__":
    unittest.main()

