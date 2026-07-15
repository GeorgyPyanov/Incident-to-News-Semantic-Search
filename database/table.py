"""SQLAlchemy table definitions for incident diagnosis storage."""

import uuid

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.sql import func

from database.settings import settings

Base = declarative_base()


class RawNews(Base):
    __tablename__ = "raw_news"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    source = Column(String(128), nullable=False)
    source_id = Column(UUID(as_uuid=True), ForeignKey("news_sources.id"), nullable=True)
    source_type = Column(String(32), nullable=False)
    url = Column(Text, nullable=True)
    url_hash = Column(String(64), nullable=True)
    title = Column(Text, nullable=False)
    body = Column(Text, nullable=True)
    language = Column(String(8), nullable=False, server_default="ru")
    published_at = Column(DateTime(timezone=True), nullable=True)
    fetched_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    last_seen_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    fetch_count = Column(Integer, nullable=False, server_default=text("1"))
    raw_region_hint = Column(String(128), nullable=True)
    raw_payload = Column(JSONB, nullable=False, server_default=text("'{}'::jsonb"))
    content_hash = Column(String(64), nullable=False)
    embedding = Column(Vector(settings.embedding_dim), nullable=True)
    embedding_model = Column(String(128), nullable=True)
    is_duplicate = Column(Boolean, nullable=False, server_default=text("false"))
    duplicate_of_id = Column(UUID(as_uuid=True), ForeignKey("raw_news.id"), nullable=True)
    processing_status = Column(String(32), nullable=False, server_default="new")
    processed_at = Column(DateTime(timezone=True), nullable=True)
    extraction_error = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())

    news_source = relationship("NewsSource", back_populates="raw_news")
    events = relationship("StructuredEvent", back_populates="raw_news")

    __table_args__ = (
        CheckConstraint(
            "processing_status IN ('new', 'processing', 'processed', 'duplicate', 'failed')",
            name="ck_raw_news_processing_status",
        ),
        Index(
            "uq_raw_news_source_url_hash",
            "source",
            "url_hash",
            unique=True,
            postgresql_where=text("url_hash IS NOT NULL"),
        ),
        Index("ix_raw_news_published_at_desc", text("published_at DESC NULLS LAST")),
        Index("ix_raw_news_content_hash", "content_hash"),
        Index("ix_raw_news_source_last_seen_at", "source", "last_seen_at"),
        Index("ix_raw_news_source_id_fetched_at", "source_id", "fetched_at"),
        Index(
            "ix_raw_news_embedding",
            embedding,
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_where=text("embedding IS NOT NULL"),
        ),
        Index(
            "ix_raw_news_unprocessed",
            "processing_status",
            text("published_at DESC NULLS LAST"),
            "fetched_at",
            postgresql_where=text("is_duplicate = false AND processed_at IS NULL"),
        ),
    )


class RawLog(Base):
    __tablename__ = "raw_logs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    dataset = Column(String(128), nullable=False)
    source = Column(String(128), nullable=False)
    line_number = Column(Integer, nullable=False)
    message = Column(Text, nullable=False)
    severity = Column(String(32), nullable=True)
    event_time = Column(DateTime(timezone=True), nullable=True)
    raw_payload = Column(JSONB, nullable=False, server_default=text("'{}'::jsonb"))
    content_hash = Column(String(64), nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())

    __table_args__ = (
        UniqueConstraint("dataset", "source", "line_number", name="uq_raw_logs_dataset_source_line"),
        Index("ix_raw_logs_dataset_source", "dataset", "source"),
        Index("ix_raw_logs_severity", "severity"),
        Index("ix_raw_logs_content_hash", "content_hash"),
        Index("ix_raw_logs_payload_gin", "raw_payload", postgresql_using="gin"),
    )


class NewsSource(Base):
    __tablename__ = "news_sources"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(128), nullable=False, unique=True)
    source_type = Column(String(32), nullable=False, server_default="rss")
    url = Column(Text, nullable=False)
    country = Column(String(64), nullable=False, server_default="RU")
    region = Column(String(128), nullable=True)
    city = Column(String(128), nullable=True)
    language = Column(String(8), nullable=False, server_default="ru")
    enabled = Column(Boolean, nullable=False, server_default=text("true"))
    poll_interval_seconds = Column(Integer, nullable=False, server_default=text("300"))
    last_fetched_at = Column(DateTime(timezone=True), nullable=True)
    source_metadata = Column("metadata", JSONB, nullable=False, server_default=text("'{}'::jsonb"))
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())

    raw_news = relationship("RawNews", back_populates="news_source")

    __table_args__ = (
        Index("ix_news_sources_enabled", "enabled", "source_type"),
        Index("ix_news_sources_location", "country", "region", "city"),
    )


class StructuredEvent(Base):
    __tablename__ = "structured_events"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    raw_news_id = Column(UUID(as_uuid=True), ForeignKey("raw_news.id"), nullable=True)
    event_type = Column(String(64), nullable=False)
    provider = Column(String(128), nullable=True)
    regions = Column(ARRAY(String), nullable=False, server_default=text("'{}'::text[]"))
    title = Column(Text, nullable=False)
    summary = Column(Text, nullable=True)
    event_start = Column(DateTime(timezone=True), nullable=True)
    event_end = Column(DateTime(timezone=True), nullable=True)
    published_at = Column(DateTime(timezone=True), nullable=True)
    extraction_method = Column(String(32), nullable=False)
    extraction_confidence = Column(Float, nullable=True)
    embedding = Column(Vector(settings.embedding_dim), nullable=True)
    embedding_model = Column(String(128), nullable=True)
    event_metadata = Column("metadata", JSONB, nullable=False, server_default=text("'{}'::jsonb"))
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())

    raw_news = relationship("RawNews", back_populates="events")
    retrieval_logs = relationship("RetrievalLog", back_populates="event")

    __table_args__ = (
        CheckConstraint(
            "extraction_confidence IS NULL OR (extraction_confidence >= 0 AND extraction_confidence <= 1)",
            name="ck_structured_events_extraction_confidence",
        ),
        Index("ix_structured_events_event_type", "event_type"),
        Index("ix_structured_events_provider", "provider"),
        Index("ix_structured_events_regions", "regions", postgresql_using="gin"),
        Index("ix_structured_events_published_at_desc", text("published_at DESC NULLS LAST")),
        Index("ix_structured_events_type_time", "event_type", text("published_at DESC NULLS LAST")),
        Index("ix_structured_events_metadata_gin", event_metadata, postgresql_using="gin"),
        Index(
            "ix_structured_events_embedding",
            embedding,
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_where=text("embedding IS NOT NULL"),
        ),
    )


class Incident(Base):
    __tablename__ = "incidents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    external_incident_id = Column(String(128), nullable=False, unique=True)
    service = Column(String(128), nullable=True)
    time_window_start = Column(DateTime(timezone=True), nullable=False)
    time_window_end = Column(DateTime(timezone=True), nullable=False)
    raw_payload = Column(JSONB, nullable=False)
    affected_regions = Column(ARRAY(String), nullable=False, server_default=text("'{}'::text[]"))
    healthy_regions = Column(ARRAY(String), nullable=False, server_default=text("'{}'::text[]"))
    affected_providers = Column(ARRAY(String), nullable=False, server_default=text("'{}'::text[]"))
    failure_type = Column(String(64), nullable=True)
    total_checks = Column(Integer, nullable=True)
    failed_checks = Column(Integer, nullable=True)
    failure_rate = Column(Float, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())

    retrieval_logs = relationship("RetrievalLog", back_populates="incident")
    reasoning_results = relationship("ReasoningResult", back_populates="incident")

    __table_args__ = (
        CheckConstraint("time_window_end >= time_window_start", name="ck_incidents_time_window"),
        CheckConstraint("failure_rate IS NULL OR (failure_rate >= 0 AND failure_rate <= 1)", name="ck_incidents_failure_rate"),
        Index("ix_incidents_time_window", "time_window_start", "time_window_end"),
        Index("ix_incidents_failure_type", "failure_type"),
        Index("ix_incidents_affected_regions", "affected_regions", postgresql_using="gin"),
    )


class RetrievalLog(Base):
    __tablename__ = "retrieval_logs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    incident_id = Column(UUID(as_uuid=True), ForeignKey("incidents.id"), nullable=False)
    event_id = Column(UUID(as_uuid=True), ForeignKey("structured_events.id"), nullable=False)
    retrieval_stage = Column(String(32), nullable=False)
    rank_position = Column(Integer, nullable=False)
    embedding_similarity = Column(Float, nullable=True)
    time_score = Column(Float, nullable=True)
    region_score = Column(Float, nullable=True)
    provider_score = Column(Float, nullable=True)
    event_type_score = Column(Float, nullable=True)
    final_score = Column(Float, nullable=True)
    was_sent_to_llm = Column(Boolean, nullable=False, server_default=text("false"))
    llm_verdict = Column(String(32), nullable=True)
    features = Column(JSONB, nullable=False, server_default=text("'{}'::jsonb"))
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())

    incident = relationship("Incident", back_populates="retrieval_logs")
    event = relationship("StructuredEvent", back_populates="retrieval_logs")

    __table_args__ = (
        CheckConstraint("rank_position > 0", name="ck_retrieval_logs_rank_position"),
        UniqueConstraint("incident_id", "event_id", "retrieval_stage", name="uq_retrieval_logs_candidate_stage"),
        Index("ix_retrieval_logs_incident", "incident_id"),
        Index("ix_retrieval_logs_event", "event_id"),
        Index("ix_retrieval_logs_stage", "retrieval_stage"),
        Index("ix_retrieval_logs_sent_to_llm", "was_sent_to_llm"),
        Index("ix_retrieval_logs_incident_stage_rank", "incident_id", "retrieval_stage", "rank_position"),
        Index("ix_retrieval_logs_features_gin", "features", postgresql_using="gin"),
    )


class ReasoningResult(Base):
    __tablename__ = "reasoning_results"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    incident_id = Column(UUID(as_uuid=True), ForeignKey("incidents.id"), nullable=False)
    cause = Column(String(64), nullable=False)
    confidence = Column(Float, nullable=False)
    summary = Column(Text, nullable=True)
    evidence = Column(JSONB, nullable=False, server_default=text("'[]'::jsonb"))
    llm_model = Column(String(128), nullable=True)
    prompt_version = Column(String(64), nullable=True)
    raw_llm_response = Column(JSONB, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())

    incident = relationship("Incident", back_populates="reasoning_results")

    __table_args__ = (
        CheckConstraint("confidence >= 0 AND confidence <= 1", name="ck_reasoning_results_confidence"),
        Index("ix_reasoning_results_incident", "incident_id"),
    )
