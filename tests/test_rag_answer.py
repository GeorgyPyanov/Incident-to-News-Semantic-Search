from __future__ import annotations

import unittest

from retrieval.db_search import DbNewsHit
from retrieval.rag_answer import (
    AbstentionPolicy,
    AnswerDraft,
    AnswerSection,
    RagAnswerService,
    evidence_overlap,
)


class FakeSearchBackend:
    def __init__(self, hits: list[DbNewsHit]) -> None:
        self.hits = hits
        self.calls: list[tuple[str, str, int]] = []

    def search(self, query: str, mode: str, top_k: int = 10) -> list[DbNewsHit]:
        self.calls.append((query, mode, top_k))
        return self.hits[:top_k]


class FakeGenerator:
    model = "qwen2.5:3b"

    def __init__(self) -> None:
        self.calls: list[tuple[str, list[DbNewsHit], str | None]] = []

    def generate(self, query: str, hits: list[DbNewsHit], *, repair_feedback: str | None = None) -> AnswerDraft:
        self.calls.append((query, list(hits), repair_feedback))
        if repair_feedback is None:
            return AnswerDraft(
                sections=(
                    AnswerSection(label="Короткий вывод", text="Проверьте зависимость и перезапустите сервис.", citation_ids=()),
                    AnswerSection(label="Что сделать", text="Проверьте зависимость и перезапустите сервис.", citation_ids=()),
                    AnswerSection(label="Почему это связано с источниками", text="В найденном документе описан похожий сбой и восстановление.", citation_ids=()),
                )
            )
        return AnswerDraft(
            sections=(
                AnswerSection(label="Короткий вывод", text="Инцидент похож на сбой зависимости.", citation_ids=(1,)),
                AnswerSection(label="Что сделать", text="Проверьте зависимость и перезапустите сервис.", citation_ids=(1,)),
                AnswerSection(label="Почему это связано с источниками", text="В документе описан похожий сбой и восстановление.", citation_ids=(1,)),
            )
        )


class RagAnswerServiceTests(unittest.TestCase):
    def test_generates_answer_with_citations_when_documents_are_relevant(self) -> None:
        backend = FakeSearchBackend([_hit(score=0.82)])
        generator = FakeGenerator()
        service = RagAnswerService(
            backend,
            generator=generator,  # type: ignore[arg-type]
            abstention_policy=AbstentionPolicy(min_top_score=0.35, min_supporting_documents=1, min_evidence_overlap=0.10),
        )

        result = service.answer("Server outage caused by dependency crash", top_k=3)

        self.assertEqual("answered", result.status)
        self.assertIn("[1]", result.answer)
        self.assertEqual("hybrid", backend.calls[0][1])
        self.assertEqual("news-1", result.citations[0].id)
        self.assertEqual(2, len(generator.calls))
        self.assertIsNone(generator.calls[0][2])
        self.assertIsNotNone(generator.calls[1][2])

    def test_abstains_when_top_document_score_is_too_low(self) -> None:
        backend = FakeSearchBackend([_hit(score=0.12)])
        generator = FakeGenerator()
        service = RagAnswerService(
            backend,
            generator=generator,  # type: ignore[arg-type]
            abstention_policy=AbstentionPolicy(min_top_score=0.35, min_supporting_documents=1, min_evidence_overlap=0.10),
        )

        result = service.answer("Как устранить проблему с падением сервера?", top_k=3)

        self.assertEqual("abstained", result.status)
        self.assertIn("Я не знаю", result.answer)
        self.assertIn("below", result.abstention_reason or "")
        self.assertEqual([], generator.calls)

    def test_abstains_when_documents_do_not_overlap_with_query_terms(self) -> None:
        backend = FakeSearchBackend([_hit(score=0.82)])
        generator = FakeGenerator()
        service = RagAnswerService(
            backend,
            generator=generator,  # type: ignore[arg-type]
            abstention_policy=AbstentionPolicy(min_top_score=0.35, min_supporting_documents=1, min_evidence_overlap=0.30),
        )

        result = service.answer("Как устранить проблему с падением сервера?", top_k=3)

        self.assertEqual("abstained", result.status)
        self.assertIn("lexical overlap", result.abstention_reason or "")
        self.assertEqual([], generator.calls)

    def test_evidence_overlap_uses_titles_sources_and_snippets(self) -> None:
        overlap = evidence_overlap("Server dependency crash", [_hit(score=0.8)])

        self.assertGreater(overlap, 0.5)


def _hit(score: float) -> DbNewsHit:
    return DbNewsHit(
        id="news-1",
        title="Server outage caused by dependency crash",
        url="https://example.test/server-outage",
        source="Example Status",
        source_type="statuspage_incident",
        published_at="2026-07-10T00:00:00+00:00",
        score=score,
        rank=1,
        snippet="Restart workers after fixing the dependency crash.",
        method="hybrid",
    )


if __name__ == "__main__":
    unittest.main()
