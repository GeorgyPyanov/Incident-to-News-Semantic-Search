"""Text formatting helpers for incident embeddings and lexical search."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict
from datetime import date, datetime
from typing import Any

from event_extraction.schemas import IncidentData


def _normalize_date(value: date | datetime | str | None) -> str | None:
    # Normalize dates into a stable ISO-like string for text generation.
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    text = str(value).strip()
    return text or None


def build_incident_embedding_text(incident: IncidentData) -> str:
    # Use a richer prompt for semantic embedding than the raw search text.
    parts: list[str] = [incident.original_log.strip()]
    if incident.entities:
        parts.append("entities: " + ", ".join(incident.entities))
    if incident.locations:
        parts.append("locations: " + ", ".join(incident.locations))
    if incident.event_types:
        parts.append("event types: " + ", ".join(incident.event_types))
    if incident.dates:
        parts.append("dates: " + ", ".join(incident.dates))
    if incident.products:
        parts.append("products: " + ", ".join(incident.products))
    if incident.services:
        parts.append("services: " + ", ".join(incident.services))
    if incident.error_descriptions:
        parts.append("errors: " + ", ".join(incident.error_descriptions))
    return "\n".join(part for part in parts if part)


def build_incident_search_text(incident: IncidentData) -> str:
    # Flatten structured incident fields into a lexical query document.
    return " ".join(
        part
        for part in (
            incident.original_log,
            " ".join(incident.entities),
            " ".join(incident.locations),
            " ".join(incident.event_types),
            " ".join(incident.dates),
            " ".join(incident.products),
            " ".join(incident.services),
            " ".join(incident.error_descriptions),
        )
        if part
    )


def incident_to_record(incident: IncidentData, *, source_id: str | None = None, title: str | None = None) -> dict[str, Any]:
    # Keep one normalized payload that can be inserted into PostgreSQL.
    data = asdict(incident)
    data["source_id"] = source_id
    data["title"] = title or incident.original_log[:200]
    data["search_text"] = build_incident_search_text(incident)
    data["embedding_text"] = build_incident_embedding_text(incident)
    data["raw_payload"] = dict(data.pop("raw_payload", {}) or {})
    return data


def build_news_embedding_text(article: Mapping[str, Any]) -> str:
    # Include summary-like fields if they exist so the vector has enough context.
    parts = [str(article.get("title", "")).strip()]
    for key in ("summary", "content", "body"):
        value = article.get(key)
        if value:
            parts.append(str(value).strip())
    for key in ("source", "source_name"):
        value = article.get(key)
        if value:
            parts.append(f"{key}: {value}")
    return "\n".join(part for part in parts if part)


def build_news_search_text(article: Mapping[str, Any]) -> str:
    # Lexical retrieval should see the same text pool as the embedding pipeline.
    return " ".join(
        str(value).strip()
        for value in (
            article.get("title"),
            article.get("summary"),
            article.get("content"),
            article.get("body"),
            article.get("source"),
            article.get("source_name"),
        )
        if value
    )

