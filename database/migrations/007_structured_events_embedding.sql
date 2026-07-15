DROP INDEX IF EXISTS ix_structured_events_embedding;

UPDATE structured_events
SET embedding = NULL,
    embedding_model = NULL,
    updated_at = now()
WHERE embedding IS NOT NULL;

ALTER TABLE structured_events
    ALTER COLUMN embedding TYPE vector(384);

CREATE INDEX IF NOT EXISTS ix_structured_events_embedding
    ON structured_events USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64)
    WHERE embedding IS NOT NULL;
