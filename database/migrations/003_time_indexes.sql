CREATE INDEX IF NOT EXISTS ix_raw_logs_event_time
    ON raw_logs (event_time);

CREATE INDEX IF NOT EXISTS ix_raw_news_source_type_published_at
    ON raw_news (source_type, published_at DESC NULLS LAST);
