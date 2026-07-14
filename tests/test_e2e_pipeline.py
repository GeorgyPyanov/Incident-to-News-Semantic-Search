from __future__ import annotations

import unittest

from api import IncidentSearchRequest, build_search_pipeline
from retrieval.reasoning import NO_STRONG_CONNECTION
from retrieval.schemas import NewsArticle


class EndToEndPipelineTests(unittest.TestCase):
    def test_original_log_flows_through_extraction_retrieval_and_reasoning(self) -> None:
        pipeline = build_search_pipeline(
            [
                NewsArticle(
                    id="news-1",
                    title="CloudPay payments-api outage in us-east-1",
                    url="https://example.test/cloudpay",
                    source="Example News",
                    published_at="2026-07-10",
                    content="CloudPay confirmed timeout and 503 errors for payments-api in us-east-1.",
                ),
                NewsArticle(
                    id="news-2",
                    title="Retail sales rise after holiday period",
                    url="https://example.test/retail",
                    source="Example News",
                    published_at=None,
                    content="Retail sales increased without any incident context.",
                ),
            ]
        )

        response = pipeline.search(
            IncidentSearchRequest(
                original_log="2026-07-10 CloudPay payments-api timeout in us-east-1 caused 503 responses.",
                top_k=2,
            )
        )

        self.assertEqual(1, len(response.results))
        self.assertEqual("news-1", response.results[0].id)
        self.assertNotEqual(NO_STRONG_CONNECTION, response.results[0].reasoning)
        self.assertIn("payments-api", response.results[0].reasoning)
        self.assertIn("us-east-1", response.results[0].reasoning)


if __name__ == "__main__":
    unittest.main()
