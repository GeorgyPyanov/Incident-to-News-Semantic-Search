from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from evaluation.benchmark_search import run_benchmark, save_results
from evaluation.dataset import EvaluationDataset, EvaluationQuery
from retrieval.schemas import NewsArticle


CPU_ONLY_HARDWARE = {
    "system": "TestOS",
    "release": "1",
    "machine": "test-machine",
    "python_version": "3.test",
    "cpu": {"model": "Test CPU", "logical_cores": 4, "physical_cores": 2},
    "total_ram_bytes": 8 * 1024**3,
    "gpu": {"model": "No GPU", "vram_bytes": None},
}


class BenchmarkSearchTests(unittest.TestCase):
    def test_result_contains_required_structure_without_gpu(self) -> None:
        results = run_benchmark(
            _dataset(),
            searches=100,
            top_k=1,
            hardware=CPU_ONLY_HARDWARE,
            timestamp="2026-07-14T00:00:00+00:00",
            save=False,
        )

        self.assertEqual("No GPU", results["platform"]["gpu"]["model"])
        self.assertEqual(2, results["embeddings"]["count"])
        self.assertEqual(32, results["embeddings"]["dimension"])
        self.assertGreater(results["index"]["size_on_disk_bytes"], 0)
        self.assertGreater(results["index"]["size_in_memory_bytes"], 0)
        self.assertEqual(1, results["search"]["warmup_searches"])
        self.assertEqual(100, results["search"]["benchmark_searches"])
        self.assertEqual(
            {"mean", "median", "p95", "min", "max"},
            set(results["search"]["latency_ms"]),
        )

    def test_saves_results_to_requested_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "nested" / "benchmark_results.json"
            results = run_benchmark(
                _dataset(),
                output_path=output,
                searches=100,
                top_k=1,
                hardware=CPU_ONLY_HARDWARE,
            )

            self.assertTrue(output.exists())
            self.assertEqual(results, json.loads(output.read_text(encoding="utf-8")))

    def test_save_results_creates_parent_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "new" / "results.json"
            save_results({"status": "ok"}, output)
            self.assertEqual({"status": "ok"}, json.loads(output.read_text(encoding="utf-8")))

    def test_rejects_fewer_than_one_hundred_searches(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least 100"):
            run_benchmark(_dataset(), searches=99, save=False)


def _dataset() -> EvaluationDataset:
    return EvaluationDataset(
        queries=(
            EvaluationQuery(
                id="q1",
                incident_log="payments API timeout",
                relevant_article_ids=frozenset({"n1"}),
                candidate_articles=(
                    NewsArticle(id="n1", title="Payments API outage", url="https://example.test/n1"),
                    NewsArticle(id="n2", title="Sunny weather", url="https://example.test/n2"),
                ),
            ),
        )
    )


if __name__ == "__main__":
    unittest.main()
