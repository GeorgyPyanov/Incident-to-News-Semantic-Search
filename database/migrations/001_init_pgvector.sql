CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS news_sources (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(128) NOT NULL UNIQUE,
    source_type VARCHAR(32) NOT NULL DEFAULT 'rss',
    url TEXT NOT NULL,
    country VARCHAR(64) NOT NULL DEFAULT 'RU',
    region VARCHAR(128),
    city VARCHAR(128),
    language VARCHAR(8) NOT NULL DEFAULT 'ru',
    enabled BOOLEAN NOT NULL DEFAULT true,
    poll_interval_seconds INTEGER NOT NULL DEFAULT 300,
    last_fetched_at TIMESTAMPTZ,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ix_news_sources_enabled
    ON news_sources (enabled, source_type);
CREATE INDEX IF NOT EXISTS ix_news_sources_location
    ON news_sources (country, region, city);

CREATE TABLE IF NOT EXISTS raw_news (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source VARCHAR(128) NOT NULL,
    source_id UUID REFERENCES news_sources(id),
    source_type VARCHAR(32) NOT NULL,
    url TEXT,
    url_hash VARCHAR(64),
    title TEXT NOT NULL,
    body TEXT,
    language VARCHAR(8) NOT NULL DEFAULT 'ru',
    published_at TIMESTAMPTZ,
    fetched_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_seen_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    fetch_count INTEGER NOT NULL DEFAULT 1,
    raw_region_hint VARCHAR(128),
    raw_payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    content_hash VARCHAR(64) NOT NULL,
    is_duplicate BOOLEAN NOT NULL DEFAULT false,
    duplicate_of_id UUID REFERENCES raw_news(id),
    processing_status VARCHAR(32) NOT NULL DEFAULT 'new',
    processed_at TIMESTAMPTZ,
    extraction_error TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT ck_raw_news_processing_status
        CHECK (processing_status IN ('new', 'processing', 'processed', 'duplicate', 'failed'))
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_raw_news_source_url_hash
    ON raw_news (source, url_hash)
    WHERE url_hash IS NOT NULL;

CREATE INDEX IF NOT EXISTS ix_raw_news_published_at_desc
    ON raw_news (published_at DESC NULLS LAST);
CREATE INDEX IF NOT EXISTS ix_raw_news_content_hash ON raw_news (content_hash);
CREATE INDEX IF NOT EXISTS ix_raw_news_source_last_seen_at ON raw_news (source, last_seen_at);
CREATE INDEX IF NOT EXISTS ix_raw_news_source_id_fetched_at ON raw_news (source_id, fetched_at);
CREATE INDEX IF NOT EXISTS ix_raw_news_unprocessed
    ON raw_news (processing_status, published_at DESC NULLS LAST, fetched_at)
    WHERE is_duplicate = false AND processed_at IS NULL;

CREATE TABLE IF NOT EXISTS structured_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    raw_news_id UUID REFERENCES raw_news(id),
    event_type VARCHAR(64) NOT NULL,
    provider VARCHAR(128),
    regions TEXT[] NOT NULL DEFAULT '{}'::text[],
    title TEXT NOT NULL,
    summary TEXT,
    event_start TIMESTAMPTZ,
    event_end TIMESTAMPTZ,
    published_at TIMESTAMPTZ,
    extraction_method VARCHAR(32) NOT NULL,
    extraction_confidence DOUBLE PRECISION,
    embedding vector(384),
    embedding_model VARCHAR(128),
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT ck_structured_events_extraction_confidence
        CHECK (extraction_confidence IS NULL OR (extraction_confidence >= 0 AND extraction_confidence <= 1))
);

CREATE INDEX IF NOT EXISTS ix_structured_events_event_type ON structured_events (event_type);
CREATE INDEX IF NOT EXISTS ix_structured_events_provider ON structured_events (provider);
CREATE INDEX IF NOT EXISTS ix_structured_events_regions ON structured_events USING GIN (regions);
CREATE INDEX IF NOT EXISTS ix_structured_events_published_at_desc
    ON structured_events (published_at DESC NULLS LAST);
CREATE INDEX IF NOT EXISTS ix_structured_events_type_time
    ON structured_events (event_type, published_at DESC NULLS LAST);
CREATE INDEX IF NOT EXISTS ix_structured_events_metadata_gin
    ON structured_events USING GIN (metadata);
CREATE INDEX IF NOT EXISTS ix_structured_events_embedding
    ON structured_events USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64)
    WHERE embedding IS NOT NULL;

CREATE TABLE IF NOT EXISTS incidents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    external_incident_id VARCHAR(128) NOT NULL UNIQUE,
    service VARCHAR(128),
    time_window_start TIMESTAMPTZ NOT NULL,
    time_window_end TIMESTAMPTZ NOT NULL,
    raw_payload JSONB NOT NULL,
    affected_regions TEXT[] NOT NULL DEFAULT '{}'::text[],
    healthy_regions TEXT[] NOT NULL DEFAULT '{}'::text[],
    affected_providers TEXT[] NOT NULL DEFAULT '{}'::text[],
    failure_type VARCHAR(64),
    total_checks INTEGER,
    failed_checks INTEGER,
    failure_rate DOUBLE PRECISION,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT ck_incidents_time_window CHECK (time_window_end >= time_window_start),
    CONSTRAINT ck_incidents_failure_rate CHECK (failure_rate IS NULL OR (failure_rate >= 0 AND failure_rate <= 1))
);

CREATE INDEX IF NOT EXISTS ix_incidents_time_window ON incidents (time_window_start, time_window_end);
CREATE INDEX IF NOT EXISTS ix_incidents_failure_type ON incidents (failure_type);
CREATE INDEX IF NOT EXISTS ix_incidents_affected_regions ON incidents USING GIN (affected_regions);

CREATE TABLE IF NOT EXISTS retrieval_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    incident_id UUID NOT NULL REFERENCES incidents(id),
    event_id UUID NOT NULL REFERENCES structured_events(id),
    retrieval_stage VARCHAR(32) NOT NULL,
    rank_position INTEGER NOT NULL,
    embedding_similarity DOUBLE PRECISION,
    time_score DOUBLE PRECISION,
    region_score DOUBLE PRECISION,
    provider_score DOUBLE PRECISION,
    event_type_score DOUBLE PRECISION,
    final_score DOUBLE PRECISION,
    was_sent_to_llm BOOLEAN NOT NULL DEFAULT false,
    llm_verdict VARCHAR(32),
    features JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT ck_retrieval_logs_rank_position CHECK (rank_position > 0),
    CONSTRAINT uq_retrieval_logs_candidate_stage UNIQUE (incident_id, event_id, retrieval_stage)
);

CREATE INDEX IF NOT EXISTS ix_retrieval_logs_incident ON retrieval_logs (incident_id);
CREATE INDEX IF NOT EXISTS ix_retrieval_logs_event ON retrieval_logs (event_id);
CREATE INDEX IF NOT EXISTS ix_retrieval_logs_stage ON retrieval_logs (retrieval_stage);
CREATE INDEX IF NOT EXISTS ix_retrieval_logs_sent_to_llm ON retrieval_logs (was_sent_to_llm);
CREATE INDEX IF NOT EXISTS ix_retrieval_logs_incident_stage_rank
    ON retrieval_logs (incident_id, retrieval_stage, rank_position);
CREATE INDEX IF NOT EXISTS ix_retrieval_logs_features_gin
    ON retrieval_logs USING GIN (features);

CREATE TABLE IF NOT EXISTS reasoning_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    incident_id UUID NOT NULL REFERENCES incidents(id),
    cause VARCHAR(64) NOT NULL,
    confidence DOUBLE PRECISION NOT NULL,
    summary TEXT,
    evidence JSONB NOT NULL DEFAULT '[]'::jsonb,
    llm_model VARCHAR(128),
    prompt_version VARCHAR(64),
    raw_llm_response JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT ck_reasoning_results_confidence CHECK (confidence >= 0 AND confidence <= 1)
);

CREATE INDEX IF NOT EXISTS ix_reasoning_results_incident ON reasoning_results (incident_id);
