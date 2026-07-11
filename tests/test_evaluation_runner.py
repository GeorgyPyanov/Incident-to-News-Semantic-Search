from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from evaluation.dataset import EvaluationDataset, EvaluationQuery
from evaluation.results import format_comparison_table
from evaluation.runner import run_evaluation
from evaluation.retrievers import default_retrieval_approaches
from retrieval.schemas import NewsArticle


class StepClock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        self.value += 0.001
        return self.value


class EvaluationRunnerTests(unittest.TestCase):
    def test_evaluates_multiple_retrieval_approaches_on_same_dataset(self) -> None:
        results = run_evaluation(
            dataset=_dataset(),
            approaches=default_retrieval_approaches(),
            timestamp="2026-07-11T00:00:00+00:00",
            clock=StepClock(),
            save=False,
        )

        self.assertEqual(
            ["keyword_lexical", "semantic_embedding", "hybrid", "current_default"],
            [result["approach_name"] for result in results],
        )
        self.assertTrue(all(result["num_queries"] == 2 for result in results))
        self.assertTrue(all(result["selected_k_values"] == [1, 3, 5] for result in results))
        self.assertIn("precision@1", results[0]["metrics"])

    def test_saves_json_and_csv_result_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            results = run_evaluation(
                dataset=_dataset(),
                output_dir=temp_dir,
                timestamp="2026-07-11T00:00:00+00:00",
                clock=StepClock(),
            )

            json_path = Path(temp_dir) / "results.json"
            csv_path = Path(temp_dir) / "results.csv"
            self.assertTrue(json_path.exists())
            self.assertTrue(csv_path.exists())

            with json_path.open("r", encoding="utf-8") as json_file:
                payload = json.load(json_file)
            self.assertEqual(len(results), len(payload["results"]))

            with csv_path.open("r", encoding="utf-8", newline="") as csv_file:
                rows = list(csv.DictReader(csv_file))
            self.assertEqual(len(results), len(rows))
            self.assertIn("approach_name", rows[0])
            self.assertIn("precision@1", rows[0])

    def test_repeated_runs_are_deterministic(self) -> None:
        first = run_evaluation(
            dataset=_dataset(),
            timestamp="2026-07-11T00:00:00+00:00",
            clock=StepClock(),
            save=False,
        )
        second = run_evaluation(
            dataset=_dataset(),
            timestamp="2026-07-11T00:00:00+00:00",
            clock=StepClock(),
            save=False,
        )

        self.assertEqual(first, second)

    def test_formats_readable_comparison_table(self) -> None:
        results = run_evaluation(
            dataset=_dataset(),
            timestamp="2026-07-11T00:00:00+00:00",
            clock=StepClock(),
            save=False,
        )

        table = format_comparison_table(results)

        self.assertIn("Approach", table)
        self.assertIn("keyword_lexical", table)
        self.assertIn("current_default", table)


def _dataset() -> EvaluationDataset:
    return EvaluationDataset(
        queries=(
            EvaluationQuery(
                id="q1",
                incident_log="CloudPay payments-api timeout in us-east-1 with 503 errors",
                relevant_article_ids=frozenset({"n1"}),
                candidate_articles=(
                    NewsArticle(
                        id="n1",
                        title="CloudPay payments-api outage in us-east-1",
                        url="https://example.test/n1",
                        content="Timeout and 503 errors affected checkout traffic.",
                    ),
                    NewsArticle(
                        id="n2",
                        title="Retail hiring report",
                        url="https://example.test/n2",
                        content="Hiring increased in stores.",
                    ),
                ),
            ),
            EvaluationQuery(
                id="q2",
                incident_log="No known external incident for internal test traffic",
                relevant_article_ids=frozenset(),
                candidate_articles=(
                    NewsArticle(
                        id="n3",
                        title="Unrelated market news",
                        url="https://example.test/n3",
                        content="Market activity was quiet.",
                    ),
                    NewsArticle(
                        id="n4",
                        title="Weather update",
                        url="https://example.test/n4",
                        content="Clear skies were expected.",
                    ),
                ),
            ),
        )
    )


if __name__ == "__main__":
    unittest.main()

