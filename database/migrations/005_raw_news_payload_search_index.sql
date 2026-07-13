CREATE INDEX IF NOT EXISTS ix_raw_news_english_payload_search
    ON raw_news
    USING GIN (
        to_tsvector(
            'english',
            coalesce(source, '') || ' ' ||
            coalesce(title, '') || ' ' ||
            coalesce(body, '') || ' ' ||
            raw_payload::text
        )
    );
