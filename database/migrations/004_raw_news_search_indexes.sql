CREATE INDEX IF NOT EXISTS ix_raw_news_english_search
    ON raw_news
    USING GIN (to_tsvector('english', coalesce(title, '') || ' ' || coalesce(body, '')));

CREATE INDEX IF NOT EXISTS ix_raw_news_source_type_time_id
    ON raw_news (source_type, published_at DESC NULLS LAST, id);
