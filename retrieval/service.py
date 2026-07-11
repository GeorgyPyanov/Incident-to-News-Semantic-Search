from __future__ import annotations

import re
from collections.abc import Sequence

from event_extraction.schemas import IncidentData
from retrieval.schemas import NewsArticle, RetrievedNewsResult


TOKEN_PATTERN = re.compile(r"[a-z0-9][a-z0-9_-]{2,}", re.IGNORECASE)


class InMemoryNewsRetriever:
    """Simple retriever implementation for local use and tests."""

    def __init__(self, articles: Sequence[NewsArticle] | None = None) -> None:
        self._articles = tuple(articles or ())

    def search(self, incident: IncidentData | str, top_k: int = 5) -> list[RetrievedNewsResult]:
        query_text = incident.original_log if isinstance(incident, IncidentData) else incident
        query_tokens = _tokens(query_text)

        ranked: list[RetrievedNewsResult] = []
        for article in self._articles:
            article_tokens = _tokens(_article_text(article))
            overlap = query_tokens & article_tokens
            if overlap:
                score = len(overlap) / max(len(query_tokens), 1)
                ranked.append(RetrievedNewsResult(article=article, score=score))

        ranked.sort(key=lambda result: result.score, reverse=True)
        return ranked[:top_k]


def _tokens(text: str | None) -> set[str]:
    return {match.group(0).lower() for match in TOKEN_PATTERN.finditer(text or "")}


def _article_text(article: NewsArticle) -> str:
    return " ".join(
        part
        for part in (
            article.title,
            article.source,
            article.published_at,
            article.content,
        )
        if part
    )
