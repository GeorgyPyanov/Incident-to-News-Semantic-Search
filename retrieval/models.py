"""Typed result objects used by the retrieval utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class SearchQuery:
    text: str
    top_k: int = 10
    filters: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SearchHit:
    id: int
    source_id: str | None
    title: str
    score: float
    rank: int
    snippet: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RankedCandidate:
    source: str
    id: int
    score: float
    rank: int
    payload: dict[str, Any] = field(default_factory=dict)

