from __future__ import annotations

import unittest

from retrieval.db_search import DbNewsHit
from retrieval.multistage import MultiStageNewsSearch


class FakeBackend:
    def search_bm25(self, query: str, top_k: int = 10) -> list[DbNewsHit]:
        return [
            self._hit("news-irrelevant", "Retail sales rise", "Example", "google_news_story", 0.95, 1),
            self._hit("news-relevant", "CloudPay payments-api outage in us-east-1", "CloudPay", "statuspage_incident", 0.90, 2),
        ][:top_k]

    def search_dense(self, query: str, top_k: int = 10, pool_size: int = 200) -> list[DbNewsHit]:
        return [
            self._hit("news-relevant", "CloudPay payments-api outage in us-east-1", "CloudPay", "statuspage_incident", 0.97, 1),
            self._hit("news-irrelevant", "Retail sales rise", "Example", "google_news_story", 0.40, 2),
        ][:top_k]

    def search_pgvector(self, query: str, top_k: int = 10) -> list[DbNewsHit]:
        return [
            self._hit("news-relevant", "CloudPay payments-api outage in us-east-1", "CloudPay", "statuspage_incident", 0.96, 1),
        ][:top_k]

    def _hit(
        self,
        hit_id: str,
        title: str,
        source: str,
        source_type: str,
        score: float,
        rank: int,
    ) -> DbNewsHit:
        return DbNewsHit(
            id=hit_id,
            title=title,
            url=f"https://example.test/{hit_id}",
            source=source,
            source_type=source_type,
            published_at="2026-07-10",
            score=score,
            rank=rank,
            snippet=title,
            method="bm25",
        )


class MultiStageNewsSearchTests(unittest.TestCase):
    def test_reranks_relevant_incident_first(self) -> None:
        service = MultiStageNewsSearch(FakeBackend())
        results = service.search("2026-07-10 CloudPay payments-api timeout in us-east-1 caused 503 responses.", top_k=2)

        self.assertEqual("news-relevant", results[0].id)
        self.assertEqual("hybrid", results[0].method)
        self.assertGreater(results[0].score, results[1].score)

    def test_normalized_sum_fusion_mode_is_available(self) -> None:
        service = MultiStageNewsSearch(FakeBackend(), fusion_mode="normalized_sum")
        results = service.search("2026-07-10 CloudPay payments-api timeout in us-east-1 caused 503 responses.", top_k=2)

        self.assertEqual(2, len(results))
        self.assertEqual("hybrid", results[0].method)

    def test_rejects_unknown_fusion_mode(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported fusion mode"):
            MultiStageNewsSearch(FakeBackend(), fusion_mode="unknown")


if __name__ == "__main__":
    unittest.main()
