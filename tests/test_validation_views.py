from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from evaluation.build_validation_views import build_views


class ValidationViewsTest(unittest.TestCase):
    def test_builds_blind_view_and_qrels(self) -> None:
        payload = {
            "version": 1,
            "examples": [
                {
                    "id": "q1",
                    "query": {
                        "log_id": "log-1",
                        "dataset": "osv_advisories",
                        "source": "pkg",
                        "message": "Go/pkg affected by GHSA-abcd-efgh-ijkl: pkg: parser vulnerability",
                        "event_time": None,
                    },
                    "relevant_news": [
                        {
                            "news_id": "doc-pos",
                            "source_type": "osv_advisory",
                            "source": "pkg",
                            "title": "pkg parser vulnerability",
                            "url": None,
                            "published_at": None,
                            "relevance_reason": "same_osv_advisory_id",
                        }
                    ],
                    "negative_news": [
                        {
                            "news_id": "doc-neg",
                            "source_type": "osv_advisory",
                            "source": "pkg",
                            "title": "pkg unrelated issue",
                            "url": None,
                            "published_at": None,
                            "negative_reason": "same_source_type_different_linkage",
                        }
                    ],
                }
            ],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "validation_set.json"
            linked = root / "validation_linked.json"
            blind = root / "validation_blind.json"
            qrels = root / "qrels.jsonl"
            source.write_text(json.dumps(payload), encoding="utf-8")

            build_views(source, linked, blind, qrels)

            blind_payload = json.loads(blind.read_text(encoding="utf-8"))
            self.assertNotIn("GHSA-abcd-efgh-ijkl", blind_payload["examples"][0]["query"]["message"])
            rows = [json.loads(line) for line in qrels.read_text(encoding="utf-8").splitlines()]
            self.assertEqual({"doc-pos": 3, "doc-neg": 1}, {row["doc_id"]: row["relevance"] for row in rows})


if __name__ == "__main__":
    unittest.main()
