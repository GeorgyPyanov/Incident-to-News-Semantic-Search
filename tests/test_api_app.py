from __future__ import annotations

import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from api.app import app
from retrieval.db_search import DbNewsHit


class ApiAppTest(unittest.TestCase):
    def test_search_endpoints_return_results(self) -> None:
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

        with patch("api.app.DbNewsSearchService") as service_cls:
            service_cls.return_value.search.return_value = [fake_hit]
            client = TestClient(app)

            for path in ("/search/bm25", "/search/dense", "/search/pgvector", "/search/hybrid"):
                response = client.post(path, json={"log": "Example provider outage", "top_k": 1})

                self.assertEqual(response.status_code, 200)
                payload = response.json()
                self.assertEqual(payload["results"][0]["id"], "news-1")
                self.assertEqual(payload["results"][0]["title"], "Provider outage resolved")


if __name__ == "__main__":
    unittest.main()
