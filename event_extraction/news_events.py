from __future__ import annotations

from datetime import datetime
from typing import Any

from database.schemas import StructuredEventCreate


OUTAGE_TERMS = ("outage", "incident", "degradation", "disruption", "unavailable", "down")
SECURITY_TERMS = ("vulnerability", "cve", "security", "advisory", "malicious")
RELEASE_TERMS = ("release", "released", "version", "tag")


class NewsStructuredEventExtractor:
    """Rule-based extractor from raw news rows to structured event records."""

    def extract(self, row: dict[str, Any]) -> StructuredEventCreate:
        source_type = row.get("source_type") or "unknown"
        payload = row.get("raw_payload") or {}
        title = row.get("title") or ""
        body = row.get("body") or ""
        text = f"{title} {body}".lower()

        provider = payload.get("provider") or payload.get("package") or row.get("source")
        if source_type == "osv_advisory":
            event_type = "security_advisory"
        elif source_type == "github_release":
            event_type = "software_release"
        elif source_type == "statuspage_incident":
            event_type = "provider_outage"
        elif any(term in text for term in SECURITY_TERMS):
            event_type = "security_advisory"
        elif any(term in text for term in RELEASE_TERMS):
            event_type = "software_release"
        elif any(term in text for term in OUTAGE_TERMS):
            event_type = "provider_outage"
        else:
            event_type = "unknown"

        event_start = _parse_datetime(payload.get("started_at") or payload.get("created_at"))
        event_end = _parse_datetime(payload.get("resolved_at") or payload.get("updated_at"))
        published_at = row.get("published_at")

        return StructuredEventCreate(
            raw_news_id=row.get("id"),
            event_type=event_type,
            provider=str(provider) if provider else None,
            regions=[],
            title=title,
            summary=_summary(body),
            event_start=event_start,
            event_end=event_end,
            published_at=published_at,
            extraction_method="rules",
            extraction_confidence=_confidence(source_type, event_type),
            metadata={
                "source": row.get("source"),
                "source_type": source_type,
                "url": row.get("url"),
                "raw_payload_keys": sorted(payload.keys())[:50],
            },
        )


def _summary(body: str | None, max_len: int = 500) -> str | None:
    if not body:
        return None
    compact = " ".join(body.split())
    if len(compact) <= max_len:
        return compact
    return compact[: max_len - 3].rstrip() + "..."


def _parse_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if not value:
        return None
    text = str(value).replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _confidence(source_type: str, event_type: str) -> float:
    if source_type in {"statuspage_incident", "osv_advisory", "github_release"}:
        return 0.95
    if event_type != "unknown":
        return 0.75
    return 0.35
