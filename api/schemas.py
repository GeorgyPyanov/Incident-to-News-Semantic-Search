from __future__ import annotations

from dataclasses import dataclass

from event_extraction.schemas import IncidentData


@dataclass(frozen=True)
class IncidentSearchRequest:
    original_log: str | None = None
    incident: IncidentData | None = None
    top_k: int = 5


@dataclass(frozen=True)
class NewsResultResponse:
    id: str
    title: str
    url: str
    source: str | None
    published_at: str | None
    score: float
    reasoning: str


@dataclass(frozen=True)
class IncidentSearchResponse:
    results: list[NewsResultResponse]
