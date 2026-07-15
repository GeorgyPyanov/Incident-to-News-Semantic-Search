from __future__ import annotations

import unittest
import importlib
from unittest.mock import patch

from fastapi.testclient import TestClient

from retrieval.db_search import DbNewsHit


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
        self.assertIn("benchmark_search", payload["files"])
        self.assertIn("benchmark_real", payload["files"])


if __name__ == "__main__":
    unittest.main()
