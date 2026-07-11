from __future__ import annotations

import unittest

from api import IncidentSearchRequest, build_search_pipeline
from retrieval.reasoning import NO_STRONG_CONNECTION
from retrieval.schemas import NewsArticle


class IncidentNewsSearchPipelineTests(unittest.TestCase):
    def test_includes_reasoning_in_each_retrieved_news_result(self) -> None:
        pipeline = build_search_pipeline(
            [
                NewsArticle(
                    id="news-1",
                    title="CloudPay payments-api outage in us-east-1",
                    url="https://example.test/cloudpay",
                    source="Example News",
                    published_at="2026-07-10",
                    content="CloudPay confirmed timeout and 503 errors for payments-api in us-east-1.",
                )
            ]
        )

        response = pipeline.search(
            IncidentSearchRequest(
                original_log="2026-07-10 CloudPay payments-api timeout in us-east-1 caused 503 responses.",
                top_k=1,
            )
        )

        self.assertEqual(1, len(response.results))
        result = response.results[0]
        self.assertEqual("news-1", result.id)
        self.assertEqual("CloudPay payments-api outage in us-east-1", result.title)
        self.assertTrue(result.reasoning)
        self.assertNotEqual(NO_STRONG_CONNECTION, result.reasoning)


if __name__ == "__main__":
    unittest.main()
