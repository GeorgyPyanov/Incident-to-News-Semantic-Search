from __future__ import annotations

import unittest

from data.embed_raw_news import embed_raw_news, raw_news_document_text


class EmbedRawNewsTests(unittest.TestCase):
    def test_raw_news_document_text_serializes_core_fields(self) -> None:
        text = raw_news_document_text(
            {
                "source": "Example Source",
                "source_type": "statuspage_incident",
                "title": "Provider outage resolved",
                "body": "Mitigation completed.",
                "raw_region_hint": "EU",
                "published_at": "2026-07-15T00:00:00+00:00",
                "raw_payload": {"id": 1, "status": "resolved"},
            }
        )

        self.assertIn("Example Source", text)
        self.assertIn("Provider outage resolved", text)
        self.assertIn("Mitigation completed.", text)
        self.assertIn('"status": "resolved"', text)

    def test_embed_raw_news_returns_zero_without_work_when_limit_is_zero(self) -> None:
        self.assertEqual(0, embed_raw_news(limit=0))

    def test_embed_raw_news_rejects_non_positive_batch_size(self) -> None:
        with self.assertRaisesRegex(ValueError, "batch_size must be positive"):
            embed_raw_news(limit=1, batch_size=0)


if __name__ == "__main__":
    unittest.main()
