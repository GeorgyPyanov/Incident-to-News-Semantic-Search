from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NewsArticle:
    id: str
    title: str
    url: str
    source: str | None = None
    published_at: str | None = None
    content: str | None = None


@dataclass(frozen=True)
class RetrievedNewsResult:
    article: NewsArticle
    score: float
