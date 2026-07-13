from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field, HttpUrl


FailureType = Literal[
    "dns_failure",
    "tls_failure",
    "http_timeout",
    "http_5xx",
    "http_4xx",
    "connection_refused",
    "unknown",
]

EventType = Literal[
    "provider_outage",
    "routing_issue",
    "dns_issue",
    "datacenter_issue",
    "power_outage",
    "government_restriction",
    "weather_event",
    "traffic_disruption",
    "unknown",
]

ExtractionMethod = Literal["rules", "gliner", "llm_fallback", "manual"]
RetrievalStage = Literal["l0_vector", "l1_rules", "l1_ranker", "l2_llm"]


class RawNewsCreate(BaseModel):
    source: str
    source_id: UUID | None = None
    source_type: str = "rss"
    url: HttpUrl | str | None = None
    url_hash: str | None = None
    title: str
    body: str | None = None
    language: str = "ru"
    published_at: datetime | None = None
    fetched_at: datetime | None = None
    last_seen_at: datetime | None = None
    raw_region_hint: str | None = None
    raw_payload: dict[str, Any] = Field(default_factory=dict)
    content_hash: str


class NewsSourceCreate(BaseModel):
    name: str
    source_type: str = "rss"
    url: HttpUrl | str
    country: str = "RU"
    region: str | None = None
    city: str | None = None
    language: str = "ru"
    enabled: bool = True
    poll_interval_seconds: int = Field(default=300, ge=60)
    metadata: dict[str, Any] = Field(default_factory=dict)


class StructuredEventCreate(BaseModel):
    raw_news_id: UUID | None = None
    event_type: EventType | str
    provider: str | None = None
    regions: list[str] = Field(default_factory=list)
    title: str
    summary: str | None = None
    event_start: datetime | None = None
    event_end: datetime | None = None
    published_at: datetime | None = None
    extraction_method: ExtractionMethod | str
    extraction_confidence: float | None = Field(default=None, ge=0, le=1)
    embedding: list[float] | None = None
    embedding_model: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class IncidentCreate(BaseModel):
    external_incident_id: str
    service: str | None = None
    time_window_start: datetime
    time_window_end: datetime
    raw_payload: dict[str, Any]
    affected_regions: list[str] = Field(default_factory=list)
    healthy_regions: list[str] = Field(default_factory=list)
    affected_providers: list[str] = Field(default_factory=list)
    failure_type: FailureType | str | None = None
    total_checks: int | None = Field(default=None, ge=0)
    failed_checks: int | None = Field(default=None, ge=0)
    failure_rate: float | None = Field(default=None, ge=0, le=1)


class RetrievalLogCreate(BaseModel):
    incident_id: UUID
    event_id: UUID
    retrieval_stage: RetrievalStage | str
    rank_position: int = Field(ge=1)
    embedding_similarity: float | None = None
    time_score: float | None = None
    region_score: float | None = None
    provider_score: float | None = None
    event_type_score: float | None = None
    final_score: float | None = None
    was_sent_to_llm: bool = False
    llm_verdict: str | None = None
    features: dict[str, Any] = Field(default_factory=dict)


class ReasoningResultCreate(BaseModel):
    incident_id: UUID
    cause: str
    confidence: float = Field(ge=0, le=1)
    summary: str | None = None
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    llm_model: str | None = None
    prompt_version: str | None = None
    raw_llm_response: dict[str, Any] | None = None
