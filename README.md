# Incident-to-News Semantic Search

This project is a search system for incident logs. A user provides an
operational log, outage update, security advisory event, or open-source project
activity record, and the system returns related news-like documents: statuspage
incident reports, OSV advisories, GitHub releases, Hacker News/Google News
stories, and GDELT events.

The target user is an SRE, support engineer, security analyst, or course
evaluator who needs to connect a short technical event to relevant external
context.

## Architecture

The system has the four required search components:

- vectorization model: local sentence-transformer embeddings for structured
  events
- vector storage: PostgreSQL with pgvector
- index: HNSW over `structured_events.embedding`
- search algorithm: BM25, dense pgvector retrieval, reciprocal-rank fusion,
  heuristic scoring, and optional LLM reranking (DeepSeek)

The production query path is:

1. parse the incident/log into structured hints
2. rewrite the query for lexical and semantic retrieval
3. retrieve candidates with BM25
4. retrieve candidates with dense pgvector search
5. fuse candidates with weighted reciprocal-rank fusion
6. rerank the shortlist with the LLM (DeepSeek), when enabled

## Data

The current database contains:

| table | rows |
| --- | ---: |
| `raw_news` | 518,768 |
| `raw_logs` | 9,718 |
| `structured_events` | 211 |
| total counted objects | 528,739 |

Data sources:

- public Statuspage incident APIs
- OSV.dev advisories
- GH Archive events and releases
- Hacker News Algolia search
- Google News RSS
- GDELT event exports

## Evaluation

The course metrics used for quality are:

- `Precision@10`
- `Recall@10`
- `MRR@10`
- `MAP`
- `nDCG@10`

The project also tracks hit rate, hard-negative hit rate, latency, hardware, HNSW
index size, and embedding generation speed.

### Blind Qrels Validation

Full run on 150 blind queries with graded qrels. Hybrid includes LLM reranking
(DeepSeek).

| mode | nDCG@10 | MRR@10 | Recall@10 | Precision@10 | hard-neg@10 | mean ms | p95 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25 | 0.234 | 0.246 | 0.172 | 0.209 | 0.000 | 31.7 | 63.0 |
| Dense | 0.836 | 0.896 | 0.639 | 0.170 | 0.453 | 76.6 | 59.8 |
| pgvector | 0.836 | 0.896 | 0.639 | 0.170 | 0.453 | 18.1 | 34.9 |
| Hybrid + LLM | 0.840 | 0.895 | 0.663 | 0.173 | 0.460 | 7088.9 | 8907.1 |

### Linked Validation

This is a sanity check on 150 linked examples where positives are known from
source identifiers. Hybrid includes LLM reranking (DeepSeek).

| mode | hit@10 | nDCG@10 | MRR@10 | Recall@10 | mean ms | p95 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25 | 0.560 | 0.515 | 0.500 | 0.560 | 836.9 | 2422.4 |
| Dense | 1.000 | 0.961 | 0.947 | 1.000 | 151.8 | 168.4 |
| pgvector | 1.000 | 0.961 | 0.947 | 1.000 | 33.3 | 46.8 |
| Hybrid + LLM | 1.000 | 0.985 | 0.980 | 1.000 | 7937.4 | 10903.2 |

### Vector Analysis

Measured on the blind validation set:

| metric | value |
| --- | ---: |
| separation rate | 0.940 |
| mean positive cosine | 0.879 |
| mean hardest-negative cosine | 0.833 |
| mean margin | 0.045 |

### System Benchmark

Read-only PostgreSQL/pgvector benchmark using already saved embeddings.

| item | value |
| --- | --- |
| OS | Windows 10 |
| Python | 3.11.2 |
| CPU | AMD Ryzen 5 5600H with Radeon Graphics, 12 logical / 6 physical cores |
| RAM | 15.36 GiB |
| GPU | No GPU |
| embedded documents | 211 / 211 |
| embedding model | `intfloat/e5-small-v2`, 384 dimensions |
| vector index | HNSW, `ix_structured_events_embedding` |
| index size | 432 KiB |
| HNSW used | yes |
| sequential scan used | no |

Retrieval latency for pgvector:

| stage | mean ms | p95 ms |
| --- | ---: | ---: |
| query embedding | 26.7 | 31.9 |
| database/index search | 2.5 | 2.8 |
| end-to-end retrieval | 29.2 | 34.4 |

Embedding generation was benchmarked in memory on 100 existing documents, with
no database writes:

| metric | value |
| --- | ---: |
| total time | 9.748 s |
| throughput | 10.26 docs/s |
| mean per document | 97.5 ms |
| p95 per document | 150.4 ms |

## Run Locally

Install dependencies and configure `.env`:

```powershell
py -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Start PostgreSQL, apply migrations, load the large dataset, and run the API:

```powershell
$env:DATABASE_URL='postgresql+asyncpg://postgres:postgres@127.0.0.1:55432/incident_news_search'
py -m database.migrate
py -m data.import_datasets --profile large
py -m uvicorn api.app:app --reload
```

Search endpoints:

- `POST /search/bm25`
- `POST /search/dense`
- `POST /search/pgvector`
- `POST /search/hybrid`
- `GET /metrics`

## Docker

Start the database and API:

```powershell
docker compose up --build postgres api
```

Run the database check and embedding jobs through Docker profiles:

```powershell
docker compose --profile tools run --rm check
docker compose --profile tools run --rm embed
docker compose --profile tools run --rm validate_linked
docker compose --profile tools run --rm validate_qrels
docker compose --profile tools run --rm embedding_analysis
docker compose --profile tools run --rm benchmark_real
```

The API is available on `http://127.0.0.1:8000`.

## Embeddings

Embeddings are stored in PostgreSQL in `structured_events.embedding` and
`structured_events.embedding_model`. The search path uses saved embeddings.

To generate embeddings for all current structured events:

```powershell
$env:DATABASE_URL='postgresql+asyncpg://postgres:postgres@127.0.0.1:55432/incident_news_search'
py -m data.embed_structured_events --limit 1000000
```

To recompute existing embeddings after changing the model or dimension:

```powershell
py -m data.embed_structured_events --refresh --limit 1000000
```

If more raw news is imported, first extract more `structured_events`, then rerun
the embedding command.

## Checks

```powershell
py -m unittest discover -v
py -m database.check --min-total 50000
py -m evaluation.validate_qrels --top-k 10
py -m evaluation.validate_retrieval --top-k 10
py -m evaluation.embedding_analysis --backend auto
py -m evaluation.benchmark_real --benchmark-document-embeddings --embedding-sample-size 100
```

Saved reports are written under `evaluation/` and are exposed by `GET /metrics`.
