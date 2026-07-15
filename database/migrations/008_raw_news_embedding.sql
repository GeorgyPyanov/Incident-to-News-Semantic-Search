ALTER TABLE raw_news
    ADD COLUMN IF NOT EXISTS embedding vector(384),
    ADD COLUMN IF NOT EXISTS embedding_model VARCHAR(128);

DROP INDEX IF EXISTS ix_raw_news_embedding;

CREATE INDEX IF NOT EXISTS ix_raw_news_embedding
    ON raw_news USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64)
    WHERE embedding IS NOT NULL;
