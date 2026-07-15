# Incident-to-News Semantic Search

Incident-to-News Semantic Search connects short incident logs to relevant
external context: statuspage incident reports, OSV advisories, GitHub releases,
Hacker News or Google News stories, and GDELT events.

The main user is an SRE, support engineer, security analyst, or evaluator who
needs to turn a short operational event into related evidence and a cited RAG
answer.

## Architecture

The system combines lexical retrieval, local embeddings, PostgreSQL/pgvector,
hybrid ranking, optional DeepSeek reranking, and evidence-grounded answer
generation.

1. Parse the input log into structured incident hints.
2. Rewrite the query for lexical and semantic retrieval.
3. Retrieve candidates with PostgreSQL BM25-style full-text search.
4. Retrieve candidates with dense pgvector search over `raw_news.embedding`.
5. Fuse BM25 and dense candidates with RRF or normalized-score fusion.
6. Apply heuristic scoring over source, provider, identifiers, and time.
7. Optionally rerank the shortlist with DeepSeek.
8. For `/answer`, generate a cited response from retrieved evidence or abstain.

Primary components:

| component | implementation |
| --- | --- |
| API | FastAPI |
| database | PostgreSQL 16 + pgvector |
| vector index | HNSW, `ix_raw_news_embedding` |
| dense corpus | `raw_news` |
| default embedding model | `intfloat/e5-small-v2`, 384 dimensions |
| fallback embedding backend | deterministic `hashing-vectorizer-384` |
| hybrid fusion | `rrf` or `normalized_sum` |
| LLM reranker | DeepSeek, optional |
| RAG generator | DeepSeek by default, Ollama optional for local generation |

For `top_k=10`, the hybrid retriever requests 120 candidates from each first
stage: original BM25, rewritten BM25, and dense pgvector. The fused pool can
contain up to 360 unique candidates. Heuristics score the full fused pool, then
the top 20 candidates form the LLM shortlist; `DEEPSEEK_RERANK_TOP_N` caps the
provider payload, and the final endpoint returns 10 results.

## Data

The local large dataset contains:

| table | rows |
| --- | ---: |
| `raw_news` | 518,768 |
| `raw_logs` | 9,718 |
| `structured_events` | 211 |
| total counted objects | 528,739 |

`raw_news` is the primary retrieval corpus. `structured_events` is an auxiliary
table for event extraction, diagnostics, and validation links.

Data sources:

- public Statuspage incident APIs
- OSV.dev advisories
- GH Archive events and GitHub releases
- Hacker News Algolia search
- Google News RSS
- GDELT event exports

## RAG Answering

`POST /answer` is a RAG endpoint: it retrieves `raw_news` evidence first, then
asks the generator to answer only from the retrieved titles, snippets, URLs, and
metadata. If the evidence is weak, the endpoint abstains instead of inventing a
fix.

Models used:

- embeddings: `intfloat/e5-small-v2` by default, with `hashing-vectorizer-384`
  as a deterministic local benchmark backend
- reranking: optional DeepSeek rerank over the hybrid shortlist
- answer generation: DeepSeek by default; Ollama `qwen2.5:3b` can be used
  locally with `RAG_GENERATOR_PROVIDER=ollama`

Response statuses:

| status | meaning |
| --- | --- |
| `answered` | evidence passed the gates and the LLM returned a valid cited answer |
| `abstained` | evidence was missing, too weak, or the generated draft failed validation |
| `llm_unavailable` | evidence was found, but the configured generator call failed |

Example:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/answer `
  -Method Post `
  -ContentType 'application/json' `
  -Body '{"log":"Cloud provider outage caused API 503 errors in us-east-1","top_k":5}'
```

## Evaluation

Reported retrieval metrics:

- `Precision@10`
- `Recall@10`
- `MRR@10`
- `nDCG@10`
- `hit@10` for linked validation
- mean, p50, and p95 latency

### Blind Qrels Validation

Evaluation on 150 blind queries with graded relevance judgments.

| mode | nDCG@10 | MRR@10 | Recall@10 | Precision@10 | mean ms | p95 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25 | 0.234 | 0.246 | 0.172 | 0.209 | 69.5 | 115.2 |
| Dense | 0.248 | 0.271 | 0.163 | 0.030 | 54.8 | 91.5 |
| pgvector | 0.248 | 0.271 | 0.163 | 0.030 | 37.9 | 49.4 |
| Hybrid + DeepSeek | 0.345 | 0.376 | 0.226 | 0.040 | 7210.1 | 9511.4 |

### Linked Validation

Sanity check on 150 linked examples where positives are known from source
identifiers.

| mode | hit@10 | nDCG@10 | MRR@10 | Recall@10 | mean ms | p95 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25 | 0.560 | 0.515 | 0.500 | 0.560 | 399.0 | 1248.0 |
| Dense | 0.367 | 0.344 | 0.337 | 0.367 | 40.0 | 116.1 |
| pgvector | 0.367 | 0.344 | 0.337 | 0.367 | 16.6 | 28.9 |
| Hybrid + DeepSeek | 0.760 | 0.724 | 0.712 | 0.760 | 8258.4 | 11053.2 |

### Vector Analysis

Measured on the blind validation set:

| metric | value |
| --- | ---: |
| separation rate | 0.900 |
| mean positive cosine | 0.439 |
| mean nearest negative cosine | 0.180 |
| mean margin | 0.258 |
| effective rank | 174.0 |

### System Benchmark

Read-only PostgreSQL/pgvector benchmark over stored `raw_news` embeddings.

| item | value |
| --- | --- |
| OS | Windows 10 |
| Python | 3.11.2 |
| CPU | AMD Ryzen 5 5600H with Radeon Graphics, 12 logical / 6 physical cores |
| RAM | 15.36 GiB |
| GPU | No GPU |
| embedded documents | `518,768 / 518,768` |
| embedding coverage | `100.0%` |
| embedding model | `hashing-vectorizer-384`, 384 dimensions |
| vector index | HNSW, `ix_raw_news_embedding` |
| index size | 994 MiB |
| HNSW used | yes |
| sequential scan used | no |

Retrieval latency for pgvector:

| stage | mean ms | p95 ms |
| --- | ---: | ---: |
| query embedding | 0.265 | 0.333 |
| database/index search | 56.71 | 131.11 |
| end-to-end retrieval | 56.98 | 131.35 |

Read-only document embedding benchmark:

| metric | value |
| --- | ---: |
| sample size | 100 |
| total time | 0.0112 s |
| throughput | 8,932.32 docs/s |
| mean per document | 0.112 ms |
| p95 per document | 0.134 ms |

## Run Locally

Install dependencies and configure `.env`:

```powershell
py -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Start PostgreSQL, apply migrations, load data, and run the API:

```powershell
$env:DATABASE_URL='postgresql+asyncpg://postgres:postgres@127.0.0.1:55432/incident_news_search'
py -m database.migrate
py -m data.import_datasets --profile large
py -m data.embed_raw_news --all
py -m uvicorn api.app:app --reload
```

Search endpoints:

- `POST /search/bm25`
- `POST /search/dense`
- `POST /search/pgvector`
- `POST /search/hybrid`
- `POST /answer`
- `GET /metrics`

## Docker

Start the database and API:

```powershell
docker compose up --build postgres api
```

DeepSeek is the default answer generator when `DEEPSEEK_API_KEY` is set. For
local answer generation, set `RAG_GENERATOR_PROVIDER=ollama`, start Ollama, and
pull the lightweight model:

```powershell
docker compose --profile llm up -d ollama
docker exec incident-news-ollama ollama pull qwen2.5:3b
docker compose up --build postgres api
```

Run tool profiles:

```powershell
docker compose --profile tools run --rm check
docker compose --profile tools run --rm embed
docker compose --profile tools run --rm validate_linked
docker compose --profile tools run --rm validate_qrels
docker compose --profile tools run --rm embedding_analysis
docker compose --profile tools run --rm benchmark_real
docker compose --profile tools run --rm compare_iterations
```

## Configuration

Useful switches:

```powershell
$env:EMBEDDING_MODEL='intfloat/e5-small-v2'
$env:EMBEDDING_BACKEND='auto'
$env:EMBEDDING_QUANTIZATION='dynamic'
$env:RETRIEVAL_FUSION_MODE='normalized_sum'
$env:DEEPSEEK_RERANK_ENABLED='true'
$env:DEEPSEEK_RERANK_TOP_N='12'
$env:RAG_GENERATOR_PROVIDER='deepseek'
$env:RAG_MIN_TOP_SCORE='0.35'
$env:RAG_MIN_EVIDENCE_OVERLAP='0.12'
```

The database schema stores 384-dimensional vectors. Use a replacement embedding
model with the same output dimension, or rebuild the schema and index for a
different dimension.

## Checks

```powershell
py -m unittest discover -v
py -m database.check --min-total 50000
py -m evaluation.validate_qrels --top-k 10
py -m evaluation.validate_retrieval --top-k 10
py -m evaluation.embedding_analysis --backend auto
py -m evaluation.benchmark_real --benchmark-document-embeddings --embedding-sample-size 100
py -m evaluation.compare_iterations --top-k 10 --embedding-sample-size 100
```

Saved reports are written under `evaluation/` and exposed by `GET /metrics`.
