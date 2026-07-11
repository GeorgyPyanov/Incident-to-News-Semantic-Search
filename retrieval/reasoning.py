from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from event_extraction.schemas import IncidentData
from retrieval.schemas import NewsArticle


NO_STRONG_CONNECTION = "No strong connection could be identified."

DATE_PATTERN = re.compile(
    r"\b(?:\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4}|"
    r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\s+\d{1,2}(?:,\s*\d{4})?)\b",
    re.IGNORECASE,
)
IDENTIFIER_PATTERN = re.compile(r"\b[a-zA-Z0-9]+(?:[-_/][a-zA-Z0-9]+)+\b")
ACRONYM_PATTERN = re.compile(r"\b[A-Z][A-Z0-9]{1,}\b")
CAPITALIZED_PHRASE_PATTERN = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2}\b")

EVENT_TERMS = (
    "outage",
    "incident",
    "degradation",
    "latency",
    "timeout",
    "error",
    "failure",
    "unavailable",
    "authentication",
    "database",
    "network",
    "dns",
)
ERROR_TERMS = (
    "timeout",
    "connection refused",
    "5xx",
    "500",
    "503",
    "rate limit",
    "authentication failure",
    "permission denied",
    "unavailable",
)


@dataclass(frozen=True)
class Evidence:
    label: str
    values: tuple[str, ...]
    weight: int


class NewsReasoningService:
    """Explains why a retrieved article may be related to an incident."""

    def explain(
        self,
        incident: IncidentData | str | Mapping[str, Any] | None,
        article: NewsArticle | Mapping[str, Any] | None,
    ) -> str:
        incident_facts = _incident_facts(incident)
        article_text = _article_text(article)

        if not _has_incident_evidence(incident_facts) or not article_text.strip():
            return NO_STRONG_CONNECTION

        evidence = _matched_evidence(incident_facts, article_text.lower())
        if sum(item.weight for item in evidence) < 2:
            return NO_STRONG_CONNECTION

        return _format_evidence(evidence)


def _incident_facts(incident: IncidentData | str | Mapping[str, Any] | None) -> dict[str, tuple[str, ...] | str]:
    if incident is None:
        return _facts_from_text("")
    if isinstance(incident, IncidentData):
        return {
            "text": incident.original_log,
            "entities": incident.entities,
            "locations": incident.locations,
            "event_types": incident.event_types,
            "dates": incident.dates,
            "products": incident.products,
            "services": incident.services,
            "error_descriptions": incident.error_descriptions,
            "identifiers": (),
        }
    if isinstance(incident, Mapping):
        text = str(incident.get("original_log") or incident.get("log") or "")
        facts = _facts_from_text(text)
        return {
            **facts,
            "entities": _as_tuple(incident.get("entities")) or facts["entities"],
            "locations": _as_tuple(incident.get("locations")) or facts["locations"],
            "event_types": _as_tuple(incident.get("event_types")) or facts["event_types"],
            "dates": _as_tuple(incident.get("dates")) or facts["dates"],
            "products": _as_tuple(incident.get("products")) or facts["products"],
            "services": _as_tuple(incident.get("services")) or facts["services"],
            "error_descriptions": _as_tuple(incident.get("error_descriptions")) or facts["error_descriptions"],
        }
    return _facts_from_text(incident)


def _facts_from_text(text: str | None) -> dict[str, tuple[str, ...] | str]:
    incident_text = text or ""
    lowered = incident_text.lower()
    identifiers = tuple(_dedupe(IDENTIFIER_PATTERN.findall(incident_text)))
    acronyms = tuple(_dedupe(ACRONYM_PATTERN.findall(incident_text)))
    phrases = tuple(_dedupe(CAPITALIZED_PHRASE_PATTERN.findall(incident_text)))

    return {
        "text": incident_text,
        "entities": acronyms + phrases,
        "locations": (),
        "event_types": tuple(term for term in EVENT_TERMS if term in lowered),
        "dates": tuple(_dedupe(match.group(0) for match in DATE_PATTERN.finditer(incident_text))),
        "products": (),
        "services": identifiers,
        "error_descriptions": tuple(term for term in ERROR_TERMS if term in lowered),
        "identifiers": identifiers,
    }


def _article_text(article: NewsArticle | Mapping[str, Any] | None) -> str:
    if article is None:
        return ""
    if isinstance(article, NewsArticle):
        parts = (
            article.title,
            article.source,
            article.published_at,
            article.content,
        )
    else:
        parts = (
            article.get("title"),
            article.get("url"),
            article.get("source"),
            article.get("published_at"),
            article.get("content"),
            article.get("summary"),
            article.get("description"),
        )
    return " ".join(str(part) for part in parts if part)


def _matched_evidence(
    incident_facts: dict[str, tuple[str, ...] | str],
    article_text_lower: str,
) -> list[Evidence]:
    categories = (
        ("entities", "entities", 2),
        ("locations", "locations", 2),
        ("products", "products", 2),
        ("services", "services", 2),
        ("error_descriptions", "error details", 2),
        ("event_types", "event type", 1),
        ("dates", "date", 1),
        ("identifiers", "identifiers", 2),
    )
    evidence: list[Evidence] = []
    seen_values: set[str] = set()

    for key, label, weight in categories:
        values = _as_tuple(incident_facts.get(key))
        matches = tuple(
            value
            for value in values
            if value.lower() not in seen_values and _contains_value(article_text_lower, value)
        )
        if matches:
            seen_values.update(value.lower() for value in matches)
            evidence.append(Evidence(label=label, values=matches, weight=weight))
    return evidence


def _has_incident_evidence(incident_facts: dict[str, tuple[str, ...] | str]) -> bool:
    if str(incident_facts.get("text") or "").strip():
        return True

    return any(
        _as_tuple(incident_facts.get(key))
        for key in (
            "entities",
            "locations",
            "event_types",
            "dates",
            "products",
            "services",
            "error_descriptions",
            "identifiers",
        )
    )


def _contains_value(article_text_lower: str, value: str) -> bool:
    normalized = value.strip().lower()
    return bool(normalized) and normalized in article_text_lower


def _format_evidence(evidence: list[Evidence]) -> str:
    primary = evidence[:3]
    fragments = [f"{item.label} ({_join_values(item.values)})" for item in primary]
    return f"The incident and news article share {', '.join(fragments)}."


def _join_values(values: tuple[str, ...]) -> str:
    if len(values) <= 2:
        return " and ".join(values)
    return ", ".join(values[:2]) + f", and {values[2]}"


def _as_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value.strip(),) if value.strip() else ()
    try:
        return tuple(str(item).strip() for item in value if str(item).strip())
    except TypeError:
        text = str(value).strip()
        return (text,) if text else ()


def _dedupe(values: object) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        text = str(value).strip()
        key = text.lower()
        if text and key not in seen:
            seen.add(key)
            deduped.append(text)
    return deduped
