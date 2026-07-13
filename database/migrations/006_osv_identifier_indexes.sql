CREATE INDEX IF NOT EXISTS ix_raw_news_osv_advisory_id
    ON raw_news ((raw_payload->>'advisory_id'))
    WHERE source_type = 'osv_advisory';

CREATE INDEX IF NOT EXISTS ix_raw_news_osv_aliases_gin
    ON raw_news
    USING GIN ((raw_payload->'aliases'))
    WHERE source_type = 'osv_advisory';
