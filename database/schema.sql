-- Enable pgvector before creating vector columns or HNSW indexes.
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS incidents (
    id BIGSERIAL PRIMARY KEY,
    source_id TEXT UNIQUE,
    original_log TEXT NOT NULL DEFAULT '',
    title TEXT NOT NULL DEFAULT '',
    description TEXT,
    incident_date DATE,
    location TEXT,
    entities TEXT[] NOT NULL DEFAULT '{}',
    event_types TEXT[] NOT NULL DEFAULT '{}',
    services TEXT[] NOT NULL DEFAULT '{}',
    products TEXT[] NOT NULL DEFAULT '{}',
    error_descriptions TEXT[] NOT NULL DEFAULT '{}',
    raw_payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    search_text TEXT NOT NULL DEFAULT '',
    search_tsv tsvector GENERATED ALWAYS AS (
        to_tsvector(
            'english',
            coalesce(title, '') || ' ' ||
            coalesce(description, '') || ' ' ||
            coalesce(original_log, '') || ' ' ||
            coalesce(location, '')
        )
    ) STORED,
    embedding vector(1536),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS news_articles (
    id BIGSERIAL PRIMARY KEY,
    source_id TEXT UNIQUE,
    title TEXT NOT NULL,
    url TEXT NOT NULL,
    source TEXT,
    published_at TIMESTAMPTZ,
    content TEXT,
    raw_payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    search_text TEXT NOT NULL DEFAULT '',
    search_tsv tsvector GENERATED ALWAYS AS (
        to_tsvector(
            'english',
            coalesce(title, '') || ' ' ||
            coalesce(source, '') || ' ' ||
            coalesce(content, '')
        )
    ) STORED,
    embedding vector(1536),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- GIN indexes back the lexical baseline.
CREATE INDEX IF NOT EXISTS idx_incidents_search_tsv ON incidents USING GIN (search_tsv);
CREATE INDEX IF NOT EXISTS idx_news_articles_search_tsv ON news_articles USING GIN (search_tsv);

-- HNSW indexes accelerate cosine-distance retrieval on embeddings.
CREATE INDEX IF NOT EXISTS idx_incidents_embedding_hnsw
    ON incidents USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS idx_news_articles_embedding_hnsw
    ON news_articles USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS idx_incidents_date ON incidents (incident_date);
CREATE INDEX IF NOT EXISTS idx_news_articles_published_at ON news_articles (published_at DESC);

-- Keep updated_at current on updates.
CREATE OR REPLACE FUNCTION touch_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_incidents_touch_updated_at ON incidents;
CREATE TRIGGER trg_incidents_touch_updated_at
BEFORE UPDATE ON incidents
FOR EACH ROW EXECUTE FUNCTION touch_updated_at();

DROP TRIGGER IF EXISTS trg_news_articles_touch_updated_at ON news_articles;
CREATE TRIGGER trg_news_articles_touch_updated_at
BEFORE UPDATE ON news_articles
FOR EACH ROW EXECUTE FUNCTION touch_updated_at();

