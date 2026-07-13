from __future__ import annotations

import unittest
from uuid import uuid4

from event_extraction.news_events import NewsStructuredEventExtractor


class NewsStructuredEventExtractorTest(unittest.TestCase):
    def test_extracts_statuspage_incident(self) -> None:
        event = NewsStructuredEventExtractor().extract(
            {
                "id": uuid4(),
                "source": "Twilio",
                "source_type": "statuspage_incident",
                "title": "Twilio: SMS delivery failures",
                "body": "Customers may be experiencing SMS delivery failures.",
                "url": "https://status.example.test/incidents/1",
                "published_at": None,
                "raw_payload": {"incident_id": "abc", "started_at": "2026-07-09T00:00:00+00:00"},
            }
        )

        self.assertEqual(event.event_type, "provider_outage")
        self.assertEqual(event.provider, "Twilio")
        self.assertEqual(event.extraction_method, "rules")
        self.assertGreaterEqual(event.extraction_confidence or 0, 0.9)

    def test_extracts_osv_advisory(self) -> None:
        event = NewsStructuredEventExtractor().extract(
            {
                "id": uuid4(),
                "source": "osv.dev",
                "source_type": "osv_advisory",
                "title": "package: vulnerability in parser",
                "body": "Security advisory body.",
                "url": None,
                "published_at": None,
                "raw_payload": {"package": "package", "advisory_id": "GHSA-xxxx-yyyy-zzzz"},
            }
        )

        self.assertEqual(event.event_type, "security_advisory")
        self.assertEqual(event.provider, "package")


if __name__ == "__main__":
    unittest.main()
