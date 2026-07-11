from __future__ import annotations

import unittest

from event_extraction.schemas import IncidentData
from retrieval.reasoning import NO_STRONG_CONNECTION, NewsReasoningService
from retrieval.schemas import NewsArticle


class NewsReasoningServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.reasoner = NewsReasoningService()

    def test_explains_clearly_related_log_and_news_article(self) -> None:
        incident = IncidentData(
            original_log="2026-07-10 CloudPay payments-api timed out in us-east-1 with 503 errors.",
            entities=("CloudPay",),
            locations=("us-east-1",),
            event_types=("outage",),
            dates=("2026-07-10",),
            services=("payments-api",),
            error_descriptions=("timeout", "503"),
        )
        article = NewsArticle(
            id="news-1",
            title="CloudPay outage affects payments-api",
            url="https://example.test/cloudpay-outage",
            source="Example News",
            published_at="2026-07-10",
            content="CloudPay reported payments-api timeout and 503 errors in us-east-1.",
        )

        explanation = self.reasoner.explain(incident, article)

        self.assertNotEqual(NO_STRONG_CONNECTION, explanation)
        self.assertIn("CloudPay", explanation)
        self.assertIn("us-east-1", explanation)
        self.assertLessEqual(len(explanation.split(".")), 3)

    def test_returns_clear_message_for_weak_or_unrelated_match(self) -> None:
        incident = IncidentData(
            original_log="2026-07-10 CloudPay payments-api timed out in us-east-1.",
            dates=("2026-07-10",),
            services=("payments-api",),
            error_descriptions=("timeout",),
        )
        article = NewsArticle(
            id="news-2",
            title="Retail sales rose on 2026-07-10",
            url="https://example.test/retail-sales",
            content="The only overlap is the calendar date, with no shared service, event, or error detail.",
        )

        self.assertEqual(NO_STRONG_CONNECTION, self.reasoner.explain(incident, article))

    def test_returns_clear_message_for_missing_or_incomplete_input(self) -> None:
        article = NewsArticle(id="news-3", title="", url="https://example.test/empty")

        self.assertEqual(NO_STRONG_CONNECTION, self.reasoner.explain(None, article))
        self.assertEqual(NO_STRONG_CONNECTION, self.reasoner.explain("", article))

    def test_explains_extracted_incident_data_without_original_log(self) -> None:
        incident = IncidentData(services=("payments-api",), error_descriptions=("503",))
        article = NewsArticle(
            id="news-4",
            title="payments-api returns elevated 503 responses",
            url="https://example.test/news-4",
        )

        explanation = self.reasoner.explain(incident, article)

        self.assertNotEqual(NO_STRONG_CONNECTION, explanation)
        self.assertIn("payments-api", explanation)


if __name__ == "__main__":
    unittest.main()
