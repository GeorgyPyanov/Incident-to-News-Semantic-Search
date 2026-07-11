# Incident-to-News-Semantic-Search

A starter incident-to-news search pipeline that connects incident logs with semantically
relevant news articles and adds a concise explanation for each retrieved result.

## Data Flow

1. `event_extraction` turns an original log into typed `IncidentData`.
2. `retrieval` ranks candidate `NewsArticle` records.
3. `retrieval.reasoning.NewsReasoningService` compares the incident data with each article.
4. `api.pipeline.IncidentNewsSearchPipeline` returns the existing news result fields plus a `reasoning` field.

When the reasoner cannot find concrete shared evidence, it returns:

```text
No strong connection could be identified.
```

## Folder Structure

```text
.
+-- api/
+-- data/
+-- database/
+-- docs/
+-- evaluation/
+-- event_extraction/
+-- retrieval/
`-- tests/
```

## Setup

Clone the repository and move into the project directory:

```bash
git clone <repository-url>
cd Incident-to-News-Semantic-Search
```

Create a local environment file:

```bash
cp .env.example .env
```

Update `.env` with the API keys, database URL, and runtime settings for your environment.

## Create a Virtual Environment

On macOS or Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

## Install Dependencies

No external dependencies are required for the current unit-tested pipeline.

## Run Tests

```bash
python -m unittest discover
```

## Evaluation Dataset

The default labeled dataset is stored at:

```text
evaluation/data/evaluation_dataset.json
```

Each query contains an `incident_log`, a list of `candidate_articles`, and
`relevant_article_ids`. To prepare a new dataset, keep the same JSON structure and use
deterministic, non-production examples so local and CI runs are reproducible.

## Run Retrieval Evaluation

Run all configured retrieval approaches with:

```bash
python -m evaluation.runner
```

The evaluation compares:

- `keyword_lexical`: token-overlap lexical retrieval.
- `semantic_embedding`: deterministic hash-based token embeddings with cosine similarity.
- `hybrid`: combined lexical and semantic score.
- `current_default`: the existing `InMemoryNewsRetriever` used by the project pipeline.

Results are saved to:

```text
evaluation/results.json
evaluation/results.csv
```

The command also prints a comparison table in the console.

## Evaluation Metrics

- `Precision@k`: the fraction of the top `k` retrieved articles that are relevant.
- `Recall@k`: the fraction of all relevant articles found in the top `k`.
- `MRR`: mean reciprocal rank of the first relevant article.
- `MAP`: mean average precision across all evaluated queries.
- `nDCG@k`: ranking quality at `k`, giving more credit when relevant articles appear higher.

## Dense, Sparse, and Hybrid Retrieval

The branch now also includes a PostgreSQL-backed retrieval path for incident-to-news matching.
This is separate from the in-memory pipeline above and is meant for the later tasks around
embeddings, pgvector search, BM25 baseline search, and hybrid ranking.

What is included:

- incident text normalization for embeddings and lexical search
- OpenAI embedding generation for incident logs
- PostgreSQL schema with `vector(1536)` columns
- HNSW indexes for dense search
- GIN + `tsvector` support for BM25-style full-text search
- reciprocal rank fusion for hybrid search
- a CLI for schema setup, embedding generation, search, and benchmarking

Typical commands:

```bash
python -m retrieval.cli init-db
python -m retrieval.cli embed-incident --original-log "API timeout after deploy"
python -m retrieval.cli fulltext-search --query-text "API timeout deploy"
python -m retrieval.cli benchmark-dense --query-embedding "[0.1, 0.2, 0.3]" --limit 10
```

The benchmark command prints latency for the indexed path and for the path with index usage
disabled, plus `EXPLAIN ANALYZE` output so you can verify whether HNSW is really chosen.
