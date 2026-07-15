from __future__ import annotations

import json
import tempfile
import unittest
import importlib
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from retrieval.db_search import DbNewsHit
from retrieval.rag_answer import AnswerCitation as RagAnswerCitation
from retrieval.rag_answer import GeneratedAnswer


class ApiAppTest(unittest.TestCase):
    def test_search_endpoints_return_results(self) -> None:
        api_app = importlib.import_module("api.app")
        fake_hit = DbNewsHit(
            id="news-1",
            title="Provider outage resolved",
            url="https://example.test/news-1",
            source="Example",
            source_type="statuspage_incident",
            published_at=None,
            score=1.0,
            rank=1,
            snippet="Provider outage resolved",
            method="hybrid",
        )

        with patch.object(api_app, "_SEARCH_SERVICE") as service:
            service.search.return_value = [fake_hit]
            client = TestClient(api_app.app)

            for path in ("/search/bm25", "/search/dense", "/search/pgvector", "/search/hybrid"):
                response = client.post(path, json={"log": "Example provider outage", "top_k": 1})

                self.assertEqual(response.status_code, 200)
                payload = response.json()
                self.assertEqual(payload["results"][0]["id"], "news-1")
                self.assertEqual(payload["results"][0]["title"], "Provider outage resolved")

    def test_metrics_endpoint_returns_pipeline_summary(self) -> None:
        api_app = importlib.import_module("api.app")
        client = TestClient(api_app.app)

        response = client.get("/metrics")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("pipeline", payload)
        self.assertIn("files", payload)
        self.assertIn("stages", payload["pipeline"])
        self.assertIn("answer_generation", payload["pipeline"])
        self.assertEqual("raw_news", payload["pipeline"]["embeddings"]["corpus"]["table"])
        self.assertIn("benchmark_search", payload["files"])
        self.assertIn("benchmark_real", payload["files"])
        self.assertNotIn("coverage_warning", payload["files"]["benchmark_real"])

    def test_benchmark_metric_file_flags_incomplete_raw_news_embeddings(self) -> None:
        api_app = importlib.import_module("api.app")
        with tempfile.TemporaryDirectory() as temp_dir:
            metric_path = Path(temp_dir) / "benchmark_real_results.json"
            metric_path.write_text(
                json.dumps(
                    {
                        "database": {"table": "raw_news"},
                        "embedding": {"embedded_documents": 64, "total_documents": 518768},
                    }
                ),
                encoding="utf-8",
            )

            payload = api_app._load_metric_file("benchmark_real", metric_path)

        self.assertIsNotNone(payload)
        self.assertIn("coverage_warning", payload)
        self.assertEqual(518704, payload["coverage_warning"]["unembedded_documents"])

    def test_answer_endpoint_returns_generated_answer(self) -> None:
        api_app = importlib.import_module("api.app")
        fake_result = GeneratedAnswer(
            status="answered",
            answer="Перезапустите сервис и проверьте зависимость [1].",
            citations=(
                RagAnswerCitation(
                    id="news-1",
                    title="Provider outage resolved",
                    url="https://example.test/news-1",
                    source="Example",
                    source_type="statuspage_incident",
                    published_at=None,
                    score=0.91,
                    rank=1,
                ),
            ),
            retrieval_mode="hybrid",
            model="qwen2.5:3b",
            abstention_reason=None,
        )

        with patch.object(api_app, "_ANSWER_SERVICE") as service:
            service.answer.return_value = fake_result
            client = TestClient(api_app.app)

            response = client.post("/answer", json={"log": "Как устранить проблему с падением сервера?", "top_k": 3})

        self.assertEqual(200, response.status_code)
        payload = response.json()
        self.assertEqual("answered", payload["status"])
        self.assertEqual("qwen2.5:3b", payload["model"])
        self.assertEqual("news-1", payload["citations"][0]["id"])
        service.answer.assert_called_once_with("Как устранить проблему с падением сервера?", top_k=3)


if __name__ == "__main__":
    unittest.main()
