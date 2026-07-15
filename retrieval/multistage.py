from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Protocol, Sequence

from event_extraction.formatting import build_incident_embedding_text
from event_extraction.schemas import IncidentData
from event_extraction.service import IncidentExtractionService
from retrieval.db_search import DbNewsHit
from retrieval.llm_reranker import DeepSeekReranker
from retrieval.query_rewrite import rewrite_incident_query


TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_-]{2,}", re.IGNORECASE)
IDENTIFIER_RE = re.compile(r"\b(?:GHSA|CVE|RUSTSEC|PYSEC|GO)-[A-Za-z0-9_.-]+\b", re.IGNORECASE)
PROVIDER_HINTS = {
    "github",
    "twilio",
    "cloudflare",
    "openai",
    "discord",
    "reddit",
    "datadog",
    "atlassian",
    "vercel",
    "netlify",
    "supabase",
    "anthropic",
    "shopify",
    "zoom",
    "pypi",
    "npm",
    "golang",
}


class NewsSearchBackend(Protocol):
    def search_bm25(self, query: str, top_k: int = 10) -> list[DbNewsHit]: ...
    def search_dense(self, query: str, top_k: int = 10, pool_size: int = 200) -> list[DbNewsHit]: ...


@dataclass(slots=True)
class QueryPlan:
    original_query: str
    incident: IncidentData
    lexical_query: str
    semantic_query: str
    tokens: tuple[str, ...]
    provider_hints: tuple[str, ...]
    identifier_hints: tuple[str, ...]


@dataclass(slots=True)
class ScoredCandidate:
    hit: DbNewsHit
    rrf_score: float
    heuristic_score: float
    llm_score: float = 0.0

    @property
    def final_score(self) -> float:
        return (0.75 * self.rrf_score) + (0.20 * self.heuristic_score) + (0.05 * self.llm_score)


class MultiStageNewsSearch:
    def __init__(
        self,
        backend: NewsSearchBackend,
        extractor: IncidentExtractionService | None = None,
        reranker: DeepSeekReranker | None = None,
    ) -> None:
        self._backend = backend
        self._extractor = extractor or IncidentExtractionService()
        self._reranker = reranker or DeepSeekReranker()

    def search(self, query: str, top_k: int = 10) -> list[DbNewsHit]:
        top_k = max(1, min(top_k, 50))
        plan = self._build_query_plan(query)
        pool_size = max(50, top_k * 12)
        shortlist_size = max(5, min(20, top_k * 2))

        candidate_lists: list[tuple[str, list[DbNewsHit]]] = [
            ("bm25", self._backend.search_bm25(plan.original_query, top_k=pool_size)),
            ("bm25_rewrite", self._backend.search_bm25(plan.lexical_query, top_k=pool_size)),
            ("dense", self._backend.search_dense(plan.semantic_query, top_k=pool_size)),
        ]

        fused = self._fuse_candidates(candidate_lists)
        if not fused:
            return []

        max_rrf = max(candidate.rrf_score for candidate in fused) or 1.0
        shortlisted = sorted(fused, key=lambda candidate: candidate.rrf_score, reverse=True)[:shortlist_size]
        llm_scores = self._reranker.rerank(plan.original_query, [candidate.hit for candidate in shortlisted])
        for candidate in fused:
            candidate.rrf_score = candidate.rrf_score / max_rrf
            candidate.heuristic_score = self._heuristic_score(plan, candidate.hit)
            candidate.llm_score = llm_scores.get(candidate.hit.id, 0.0)

        ranked = sorted(fused, key=lambda candidate: (candidate.final_score, candidate.hit.score, candidate.hit.published_at or ""), reverse=True)
        return [
            DbNewsHit(
                **{
                    **candidate.hit.__dict__,
                    "score": candidate.final_score,
                    "rank": rank,
                    "method": "hybrid",
                }
            )
            for rank, candidate in enumerate(ranked[:top_k], start=1)
        ]

    def _build_query_plan(self, query: str) -> QueryPlan:
        incident = self._extractor.extract(query)
        lexical_query = " ".join(part for part in (query, rewrite_incident_query(query)) if part.strip())
        semantic_query = build_incident_embedding_text(incident) if incident.has_structured_facts else rewrite_incident_query(query)
        tokens = tuple(_tokens(query))
        provider_hints = tuple(sorted(token for token in tokens if token in PROVIDER_HINTS))
        identifier_hints = tuple(_identifiers(query))
        return QueryPlan(
            original_query=query,
            incident=incident,
            lexical_query=lexical_query,
            semantic_query=semantic_query,
            tokens=tokens,
            provider_hints=provider_hints,
            identifier_hints=identifier_hints,
        )

    def _fuse_candidates(self, candidate_lists: Sequence[tuple[str, Sequence[DbNewsHit]]], k: float = 60.0) -> list[ScoredCandidate]:
        stage_weights = {
            "bm25": 0.85,
            "bm25_rewrite": 0.80,
            "dense": 1.25,
        }
        by_id: dict[str, ScoredCandidate] = {}
        for stage_name, hits in candidate_lists:
            stage_weight = stage_weights.get(stage_name, 1.0)
            for hit in hits:
                entry = by_id.get(hit.id)
                if entry is None:
                    entry = ScoredCandidate(hit=hit, rrf_score=0.0, heuristic_score=0.0)
                    by_id[hit.id] = entry
                entry.rrf_score += stage_weight / (k + hit.rank)
                if hit.score > entry.hit.score:
                    entry.hit = hit
        return list(by_id.values())

    def _heuristic_score(self, plan: QueryPlan, hit: DbNewsHit) -> float:
        article_text = " ".join(
            part
            for part in (
                hit.title,
                hit.snippet or "",
                hit.source,
                hit.source_type,
            )
            if part
        ).lower()
        query_tokens = set(plan.tokens or _tokens(plan.semantic_query))
        article_tokens = set(_tokens(article_text))
        lexical_overlap = len(query_tokens & article_tokens) / max(1, len(query_tokens))
        identifier_match = 1.0 if any(identifier.lower() in article_text for identifier in plan.identifier_hints) else 0.0
        provider_match = 1.0 if any(provider in article_text for provider in plan.provider_hints) else 0.0
        source_prior = _source_prior(plan.incident, hit)
        recency_score = _recency_score(plan.incident, hit.published_at)
        score = (
            0.45 * lexical_overlap
            + 0.20 * identifier_match
            + 0.15 * provider_match
            + 0.10 * source_prior
            + 0.10 * recency_score
        )
        return min(1.0, score)


def _source_prior(incident: IncidentData, hit: DbNewsHit) -> float:
    lowered = " ".join(
        part.lower()
        for part in (
            incident.original_log,
            " ".join(incident.event_types),
            " ".join(incident.error_descriptions),
        )
        if part
    )
    source_type = hit.source_type.lower()

    if any(token in lowered for token in ("ghsa", "cve", "rustsec", "pysec", "vulnerability", "security", "advisory")):
        return 1.0 if source_type == "osv_advisory" else 0.25
    if "release" in lowered or "version" in lowered or "tag" in lowered:
        return 1.0 if source_type == "github_release" else 0.25
    if any(token in lowered for token in ("outage", "incident", "degradation", "timeout", "failure", "unavailable", "dns")):
        return 1.0 if source_type in {"statuspage_incident", "hackernews_story", "google_news_story", "gdeltv2_event"} else 0.35
    return 0.5 if source_type in {"statuspage_incident", "github_release", "osv_advisory"} else 0.2


def _recency_score(incident: IncidentData, published_at: str | None) -> float:
    if not incident.dates or not published_at:
        return 0.0
    query_date = _parse_date(incident.dates[0])
    article_date = _parse_date(published_at)
    if query_date is None or article_date is None:
        return 0.0
    delta_days = abs((article_date - query_date).days)
    return math.exp(-delta_days / 14.0)


def _parse_date(value: str) -> datetime | None:
    text = value.strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _tokens(text: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_RE.finditer(text or "")]


def _identifiers(text: str) -> list[str]:
    seen: set[str] = set()
    values: list[str] = []
    for match in IDENTIFIER_RE.finditer(text or ""):
        value = match.group(0)
        key = value.lower()
        if key not in seen:
            seen.add(key)
            values.append(value)
    return values
