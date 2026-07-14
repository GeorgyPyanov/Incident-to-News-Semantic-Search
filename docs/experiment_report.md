# Experiment Report

## Value Proposition

The system helps operators investigate incident/status logs by returning related
news, statuspage incident reports, GitHub releases, and open-source security
advisories. The target user is an SRE, support engineer, or analyst who has a
short operational log and needs external context quickly.

## Current Architecture

1. Load raw data into PostgreSQL:
   - `raw_logs`: status updates, OSV package events, GitHub activity
   - `raw_news`: statuspage incident reports, HN/Google stories, GDELT events,
     OSV advisories, GitHub releases
2. Extract rule-based `structured_events` from raw news.
3. Embed structured events with a local hashing vectorizer into `pgvector`.
4. Search through FastAPI endpoints:
   - `/search/bm25`: lexical baseline with query rewriting and identifier lookup
   - `/search/dense`: local dense-like candidate reranking
   - `/search/pgvector`: vector search over `structured_events.embedding`
   - `/search/hybrid`: reciprocal-rank fusion of BM25, dense, and pgvector

## Dataset

The included Docker dump contains:

- `raw_news`: 344 449
- `raw_logs`: 9 210
- `structured_events`: 211
- total counted objects: 353 912

The project now keeps two validation views over the same 150 labeled queries:

- `validation_linked.json`: source-linked sanity validation with strong labels.
- `validation_blind.json`: direct IDs and title-like query prefixes removed.
- `qrels.jsonl`: graded relevance judgments for blind evaluation.

The base set contains:

- 50 Statuspage logs linked to incident reports by `incident_id`
- 50 OSV logs linked to advisories by `advisory_id`
- 50 GH Archive logs linked to GitHub releases by repository/time window

Each validation example also includes hard negatives from the same `source_type`
with different linkage.

The graded qrels use:

- `3`: same incident/advisory/release
- `1`: topically related hard negative
- `0`: unrelated or different-linkage negative

## Iterations

### Iteration 1: Lexical Baseline

BM25-style full-text retrieval was implemented with PostgreSQL GIN indexes over
source, title, body, and raw payload. This baseline is strong when the log shares
exact identifiers or provider/status vocabulary with the relevant news.

Known weakness: GitHub release relevance is often repository/time based, not
pure lexical relevance, so BM25 alone misses many GitHub validation pairs.

### Iteration 2: Dense And Hybrid Retrieval

A dense-like hashing vectorizer was added for local deterministic search, plus a
`pgvector` path over `structured_events.embedding`. Hybrid retrieval fuses BM25,
dense, and pgvector rankings with reciprocal-rank fusion.

This improves vocabulary mismatch and repository/time cases while keeping BM25's
exact-match strengths for Statuspage and OSV identifiers.

### Iteration 3: pgvector Structured Events

Validation-relevant news records are transformed into `structured_events`,
embedded with a deterministic local hashing vectorizer, and searched through
PostgreSQL `pgvector`. This provides a real vector database path that is fast on
CPU and does not require external API keys.

The current pgvector path has high recall but can return hard negatives close to
the positives. That is expected for same-source-type negatives and motivates a
future reranking stage.

## Evaluation

Run:

```powershell
$env:DATABASE_URL='postgresql+asyncpg://postgres:postgres@127.0.0.1:55432/incident_news_search'
py -m evaluation.validate_retrieval --top-k 10
py -m evaluation.benchmark_search --top-k 10 --max-queries 60
```

Primary quality metrics:

- hit@10
- recall@10
- MRR@10
- nDCG@10
- negative-hit@10

System metrics:

- mean latency
- p50 latency
- p95 latency
- average number of returned results

Current linked-validation summary:

| Mode | hit@10 | MRR@10 | nDCG@10 | negative-hit@10 |
| --- | ---: | ---: | ---: | ---: |
| BM25 | 0.66 | 0.60 | 0.62 | 0.00 |
| Dense | 1.00 | 0.95 | 0.97 | 0.00 |
| pgvector | 0.99 | 0.95 | 0.96 | 0.52 |
| Hybrid | 1.00 | 0.96 | 0.97 | 0.49 |

Current blind qrels summary:

| Mode | nDCG@10 | MRR@10 | Recall@10 | Precision@10 | Hard-negative hit@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| BM25 | 0.29 | 0.31 | 0.20 | 0.27 | 0.00 |
| Dense | 0.52 | 0.55 | 0.37 | 0.32 | 0.13 |
| pgvector | 0.84 | 0.90 | 0.66 | 0.17 | 0.41 |
| Hybrid | 0.72 | 0.71 | 0.65 | 0.16 | 0.39 |

This blind qrels result is the more honest offline estimate. The linked result
is useful as a pipeline sanity check but overstates semantic retrieval quality
because some queries contain strong source-specific identifiers.

Current benchmark on 60 validation queries:

| Mode | mean ms | p50 ms | p95 ms |
| --- | ---: | ---: | ---: |
| BM25 | 380.5 | 49.5 | 2017.0 |
| Dense | 357.2 | 48.6 | 2826.4 |
| pgvector | 44.0 | 44.6 | 62.8 |
| Hybrid | 781.9 | 142.4 | 3065.6 |

## Hardware Requirements

Development and demo inference have been tested on a local laptop with Docker
Desktop and PostgreSQL/pgvector. The included dump is about 66 MB compressed.
The database restores into a Docker volume and serves local demo traffic on CPU.

The current deterministic dense path does not require GPU. A future neural
embedding model can improve semantic quality but will increase build time,
storage, and inference requirements.

## Recommended Next Iteration

- Generate neural embeddings for a larger `structured_events` subset using an
  open-source sentence-transformer model.
- Store chunk-level embeddings for long news bodies.
- Add a cross-encoder or lightweight reranker for the top 20 hybrid candidates.
- Add query rewriting/HyDE for short logs that lack provider-specific terms.
