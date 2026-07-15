from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Protocol, Sequence

import httpx


class RerankCandidate(Protocol):
    id: str
    title: str
    source: str
    source_type: str
    published_at: str | None
    snippet: str | None
    score: float

@dataclass(slots=True)
class DeepSeekReranker:
    model: str = field(default_factory=lambda: os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash"))
    api_key: str | None = field(default_factory=lambda: os.environ.get("DEEPSEEK_API_KEY"))
    base_url: str = field(default_factory=lambda: os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))
    enabled: bool = field(default_factory=lambda: _truthy(os.environ.get("DEEPSEEK_RERANK_ENABLED")))
    max_candidates: int = field(default_factory=lambda: int(os.environ.get("DEEPSEEK_RERANK_TOP_N", "12")))
    timeout_seconds: float = field(default_factory=lambda: float(os.environ.get("DEEPSEEK_TIMEOUT_SECONDS", "20")))
    _client: httpx.Client | None = field(init=False, default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.enabled or not self.api_key:
            self._client = None
            return
        self._client = httpx.Client(
            base_url=self.base_url.rstrip("/"),
            timeout=self.timeout_seconds,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
        )

    def rerank(self, query: str, candidates: Sequence[RerankCandidate]) -> dict[str, float]:
        if self._client is None or not candidates:
            return {}

        shortlist = list(candidates[: self.max_candidates])
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You rank news results for an incident log. "
                        "Return strict JSON only: "
                        '{"results":[{"id":"...","score":0-100,"reason":"..."}]}. '
                        "Higher score means more relevant."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "query": query,
                            "candidates": [
                                {
                                    "id": candidate.id,
                                    "title": candidate.title,
                                    "source": candidate.source,
                                    "source_type": candidate.source_type,
                                    "published_at": candidate.published_at,
                                    "snippet": candidate.snippet,
                                }
                                for candidate in shortlist
                            ],
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            "temperature": 0,
            "max_tokens": 900,
        }

        try:
            response = self._client.post("/chat/completions", json=payload)
            response.raise_for_status()
        except httpx.HTTPError:
            return {}

        try:
            body = response.json()
            content = body["choices"][0]["message"]["content"] or ""
        except (ValueError, KeyError, IndexError, TypeError):
            return {}

        return _parse_score_map(str(content), shortlist)

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None


def _parse_score_map(content: str, candidates: Sequence[RerankCandidate]) -> dict[str, float]:
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if not match:
            return {}
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}

    if not isinstance(payload, dict):
        return {}
    results = payload.get("results")
    if not isinstance(results, list):
        return {}

    candidate_ids = {candidate.id for candidate in candidates}
    scores: dict[str, float] = {}
    for item in results:
        if not isinstance(item, dict):
            continue
        candidate_id = str(item.get("id") or "").strip()
        if candidate_id not in candidate_ids:
            continue
        try:
            raw_score = float(item.get("score") or 0.0)
        except (TypeError, ValueError):
            continue
        scores[candidate_id] = max(0.0, min(1.0, raw_score / 100.0))
    return scores


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}
