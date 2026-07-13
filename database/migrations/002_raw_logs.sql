CREATE TABLE IF NOT EXISTS raw_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    dataset VARCHAR(128) NOT NULL,
    source VARCHAR(128) NOT NULL,
    line_number INTEGER NOT NULL,
    message TEXT NOT NULL,
    severity VARCHAR(32),
    event_time TIMESTAMPTZ,
    raw_payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    content_hash VARCHAR(64) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT uq_raw_logs_dataset_source_line UNIQUE (dataset, source, line_number)
);

CREATE INDEX IF NOT EXISTS ix_raw_logs_dataset_source
    ON raw_logs (dataset, source);
CREATE INDEX IF NOT EXISTS ix_raw_logs_severity
    ON raw_logs (severity);
CREATE INDEX IF NOT EXISTS ix_raw_logs_content_hash
    ON raw_logs (content_hash);
CREATE INDEX IF NOT EXISTS ix_raw_logs_payload_gin
    ON raw_logs USING GIN (raw_payload);
