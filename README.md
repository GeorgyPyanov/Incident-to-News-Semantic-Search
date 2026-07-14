# Incident-to-News-Semantic-Search

A project for connecting incident/status logs with semantically relevant news,
incident reports, project activity, and security advisories.

## Pipeline Tasks

The task-oriented pipeline connects an original log to candidate articles and
adds concise reasoning for each retrieved result.

Data flow:

1. `event_extraction` turns an original log into typed `IncidentData`.
2. `retrieval` ranks candidate `NewsArticle` records.
3. `retrieval.reasoning.NewsReasoningService` compares the incident data with each article.
4. `api.pipeline.IncidentNewsSearchPipeline` returns the existing news result fields plus a `reasoning` field.

When the reasoner cannot find concrete shared evidence, it returns:

```text
No strong connection could be identified.
```

Current folders:

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

## Database And Dataset Scope

Done for the database/data-loading stage:

- PostgreSQL schema and migrations.
- `raw_news` table for news/event records.
- `raw_logs` table for availability/status update logs.
- Dataset import script with more than 50 000 objects.
- Database count check script.
- Docker-ready compressed PostgreSQL dump.

Current dataset choice:

- Recent news/events: GDELT 2.1 Events from the latest 15-minute export files.
- Targeted tech news/discussions: Hacker News Algolia stories queried by
  provider + outage/incident/status keywords.
- Targeted real news: Google News RSS stories queried by provider +
  outage/down/incident/status keywords.
- Open-source security advisories: OSV.dev advisories for popular PyPI, npm,
  Go, Maven, and crates.io packages.
- Open-source project activity/news: GH Archive public GitHub events. Release
  events are stored as `github_release` news, and matching GitHub activity
  events are stored as logs.
- Availability/status logs: public Statuspage incident APIs for GitHub,
  Cloudflare, OpenAI, Discord, Reddit, Datadog, Atlassian, Twilio, SendGrid,
  DigitalOcean, Vercel, Netlify, Supabase, Anthropic, Shopify, and Zoom.
- Status incident reports are also stored in `raw_news` as
  `statuspage_incident`, so each update log can be joined back to its incident
  by `raw_payload.incident_id`.

The old AG News / Loghub Apache / GDELT 2005 rows were removed from the local
database because their timeline did not match the availability-log task.

Current included Docker dump:

- `raw_news`: 344 449 rows
- `raw_logs`: 9 210 rows
- `news_sources`: 42 rows
- `structured_events`: 211 rows
- total selected objects: 353 912 rows

Relevance checks:

- 7 882 status update logs have direct linked incident reports.
- 5 805 status update logs have Hacker News provider/time candidates.
- 3 938 status update logs have Google News provider/time candidates.
- 417 GitHub activity logs have matching GitHub release rows.
- 860 OSV package events have direct linked security advisories.

Open-source subset:

- 300 GH Archive projects selected
- 409 `github_release` rows
- 468 `gharchive_open_source` log rows

## Setup

```powershell
py -m pip install -r requirements.txt
Copy-Item .env.example .env
```

For the included Docker PostgreSQL:

```dotenv
DATABASE_URL=postgresql+asyncpg://postgres:postgres@127.0.0.1:55432/incident_news_search
EMBEDDING_DIM=1024
```

## Run Database

The repository includes a compressed PostgreSQL dump at
`database/initdb/001_incident_news_search.sql.gz`. On a fresh Docker volume,
PostgreSQL restores it automatically through `/docker-entrypoint-initdb.d`.

```powershell
docker compose up -d postgres
```

If you already have an old local Docker volume and need to reload the included
dump from scratch:

```powershell
docker compose down -v
docker compose up -d postgres
```

## Apply Migrations

Only run migrations when creating an empty database without the included dump.
The included dump already contains the schema and loaded data.

```powershell
$env:DATABASE_URL='postgresql+asyncpg://postgres:postgres@127.0.0.1:55432/incident_news_search'
py -m database.migrate
```

## Load Current Dataset

```powershell
$env:DATABASE_URL='postgresql+asyncpg://postgres:postgres@127.0.0.1:55432/incident_news_search'
py -m data.import_datasets --news-limit 0 --logs-limit 0 --gdelt-limit 0 --statuspage-limit 5000 --hn-limit-per-provider 100 --google-news-limit-per-provider 80 --osv-limit-per-package 100 --gdeltv2-limit 250000 --gdeltv2-start 2026-06-01 --gdeltv2-end 2026-07-07 --gdeltv2-max-files 400 --gdeltv2-max-files-per-day 8 --gharchive-projects 300 --gharchive-log-limit 50000 --gharchive-news-limit 2000
```

## Check Loaded Rows

```powershell
$env:DATABASE_URL='postgresql+asyncpg://postgres:postgres@127.0.0.1:55432/incident_news_search'
py -m database.check
```

## Check Relevance Coverage

```powershell
$env:DATABASE_URL='postgresql+asyncpg://postgres:postgres@127.0.0.1:55432/incident_news_search'
py -m database.relevance_check
```

The expensive GDELT provider/time diagnostic is optional:

```powershell
py -m database.relevance_check --include-gdelt
```

## Extract Structured Events

Convert raw news records into rule-based `structured_events` rows:

```powershell
$env:DATABASE_URL='postgresql+asyncpg://postgres:postgres@127.0.0.1:55432/incident_news_search'
py -m data.extract_structured_events --limit 1000
```

## Validation Set

The validation set is stored at:

```text
evaluation/data/validation_set.json
```

It contains 150 labeled log-news examples:

- 50 Statuspage update logs linked to incident reports by `incident_id`
- 50 OSV package logs linked to advisories by `advisory_id`
- 50 GH Archive activity logs linked to GitHub releases by repository and time window

Regenerate it from the loaded database with:

```powershell
$env:DATABASE_URL='postgresql+asyncpg://postgres:postgres@127.0.0.1:55432/incident_news_search'
py -m evaluation.build_validation_set --limit-per-source 50
```

Run retrieval validation:

```powershell
$env:DATABASE_URL='postgresql+asyncpg://postgres:postgres@127.0.0.1:55432/incident_news_search'
py -m evaluation.validate_retrieval --top-k 10
```

Current validation result on the included dump:

- `bm25`: hit@10 = 0.66
- `dense`: hit@10 = 1.00
- `pgvector`: hit@10 = 0.99
- `hybrid`: hit@10 = 1.00

## FastAPI Demo

Run the API:

```powershell
$env:DATABASE_URL='postgresql+asyncpg://postgres:postgres@127.0.0.1:55432/incident_news_search'
py -m uvicorn api.app:app --reload
```

Search endpoints:

- `POST /search/bm25`
- `POST /search/dense`
- `POST /search/hybrid`
- `POST /search/pgvector`

Example request:

```json
{
  "log": "Twilio SMS delivery failures from Twilio Phone Numbers to Spusu Italy investigating",
  "top_k": 5
}
```

## Long Real-Source Harvest

Continuously harvest paired real sources without Loghub:

```powershell
$env:DATABASE_URL='postgresql+asyncpg://postgres:postgres@127.0.0.1:55432/incident_news_search'
py -m data.harvest_real_sources --duration-hours 6 --interval-minutes 30 --statuspage-limit 5000 --hn-limit-per-provider 100 --osv-limit-per-package 20 --gdeltv2-limit 10000 --gdeltv2-max-files 24 --gharchive-projects 300 --gharchive-log-limit 50000 --gharchive-news-limit 2000 --gharchive-lookback-hours 6
```

Progress is written to `logs/harvest_real_sources.log`.

## Run Tests

```bash
python -m unittest discover -v
```

Current test coverage includes:

- `tests/test_embeddings.py` for OpenAI embedding generation behavior
- `tests/test_e2e_pipeline.py` for the end-to-end extraction -> retrieval -> reasoning path
- `tests/test_pipeline.py` and the existing evaluation and API tests for the current project flow

The full suite is expected to pass locally before changes are merged.

## Evaluation Dataset

The default labeled dataset is stored at:

```text
evaluation/data/evaluation_dataset.json
```

Each query contains an `incident_log`, a list of `candidate_articles`, and
`relevant_article_ids`. To prepare a new dataset, keep the same JSON structure
and use deterministic, non-production examples so local and CI runs are
reproducible.

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

## Benchmark Retrieval Performance

Two benchmarks are available and write to separate result files.

The synthetic benchmark uses the small three-query evaluation dataset and an
in-memory exact-cosine index. It is dependency-light and intended for local and
CI regression checks:

```bash
python -m evaluation.benchmark_search
```

The command times document embedding and exact-cosine index construction
separately, performs one warm-up search, and reports latency statistics for 100
measured searches. It also records the dataset, model, index, platform, CPU,
RAM, and GPU (or `No GPU`) details. Results are printed to the console and saved
to:

```text
evaluation/benchmark_results.json
```

Use `--searches`, `--top-k`, `--dimensions`, `--dataset`, or `--output` to
override benchmark settings. `--searches` must be at least 100.

### Real PostgreSQL/pgvector Benchmark

The real benchmark uses representative queries from the 150-example validation
set, the existing `structured_events` embeddings, the production query
vectorizer and dense-search SQL, and the existing pgvector HNSW index. Its safe
default is read-only: it neither generates embeddings nor rebuilds the index.

Requirements:

- install `requirements.txt` (including `psycopg`);
- start the included PostgreSQL/pgvector database;
- set `DATABASE_URL` and `EMBEDDING_DIM` as shown in `.env.example`;
- ensure the database migrations, structured events, embeddings, and HNSW index
  already exist.

Safe read-only benchmark command:

```powershell
$env:DATABASE_URL='postgresql+asyncpg://postgres:postgres@127.0.0.1:55432/incident_news_search'
$env:EMBEDDING_DIM='1024'
python -m evaluation.benchmark_real
```

Results are printed and saved separately to:

```text
evaluation/benchmark_real_results.json
```

Embedding generation is opt-in and only fills currently missing
`structured_events.embedding` values using the existing generator:

```powershell
python -m evaluation.benchmark_real --generate-embeddings --embedding-limit 1000
```

Rebuilding the existing production HNSW index is also opt-in:

```powershell
python -m evaluation.benchmark_real --rebuild-index
```

Both optional flags modify the database and can be combined when explicitly
needed. The real result includes PostgreSQL-reported index size and
configuration, separate query-embedding/database/end-to-end latency statistics,
and an `EXPLAIN ANALYZE` summary showing whether the HNSW index or a sequential
scan was used. The existing synthetic result remains at
`evaluation/benchmark_results.json` and is never overwritten by this command.

### Safe Document Embedding Benchmark

Benchmark the exact production `structured_events` text preparation and
`hashing-vectorizer-1024` generator on 100 existing documents, entirely in
memory:

```powershell
$env:DATABASE_URL='postgresql+asyncpg://postgres:postgres@127.0.0.1:55432/incident_news_search'
$env:EMBEDDING_DIM='1024'
python -m evaluation.benchmark_real --benchmark-document-embeddings
```

Use `--embedding-sample-size N` to change the default sample size of 100. This
mode performs one untimed warm-up, runs inside a read-only PostgreSQL
transaction, and cannot be combined with `--generate-embeddings` or
`--rebuild-index`. Vectors are generated in memory and never persisted. The
result is stored under `document_embedding_benchmark` in
`evaluation/benchmark_real_results.json`; the existing real retrieval and
synthetic benchmark commands remain unchanged.

## Evaluation Metrics

- `Precision@k`: the fraction of the top `k` retrieved articles that are relevant.
- `Recall@k`: the fraction of all relevant articles found in the top `k`.
- `MRR`: mean reciprocal rank of the first relevant article.
- `MAP`: mean average precision across all evaluated queries.
- `nDCG@k`: ranking quality at `k`, giving more credit when relevant articles appear higher.

## Dense, Sparse, And Hybrid Retrieval

The branch also includes a PostgreSQL-backed retrieval path for incident-to-news
matching. This is separate from the in-memory pipeline above and is meant for
later tasks around embeddings, pgvector search, BM25 baseline search, and hybrid
ranking.

What is included:

- incident text normalization for embeddings and lexical search
- OpenAI embedding generation for incident logs
- PostgreSQL schema with vector columns
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

The benchmark command prints latency for the indexed path and for the path with
index usage disabled, plus `EXPLAIN ANALYZE` output so you can verify whether
HNSW is really chosen.
