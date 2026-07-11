from __future__ import annotations

import re

from event_extraction.schemas import IncidentData


DATE_PATTERN = re.compile(
    r"\b(?:\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4}|"
    r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\s+\d{1,2}(?:,\s*\d{4})?)\b",
    re.IGNORECASE,
)
IDENTIFIER_PATTERN = re.compile(r"\b[a-zA-Z0-9]+(?:[-_/][a-zA-Z0-9]+)+\b")
ACRONYM_PATTERN = re.compile(r"\b[A-Z][A-Z0-9]{1,}\b")

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


class IncidentExtractionService:
    """Lightweight incident fact extractor used by the search pipeline."""

    def extract(self, original_log: str | None) -> IncidentData:
        log = original_log or ""
        lowered = log.lower()

        dates = tuple(_dedupe(match.group(0) for match in DATE_PATTERN.finditer(log)))
        identifiers = tuple(_dedupe(IDENTIFIER_PATTERN.findall(log)))
        acronyms = tuple(_dedupe(ACRONYM_PATTERN.findall(log)))
        event_types = tuple(term for term in EVENT_TERMS if term in lowered)
        errors = tuple(term for term in ERROR_TERMS if term in lowered)

        return IncidentData(
            original_log=log,
            entities=acronyms,
            event_types=event_types,
            dates=dates,
            services=identifiers,
            error_descriptions=errors,
        )


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
