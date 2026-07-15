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
3. Embed structured events with the configured local encoder into `pgvector`.
   The default learned model is `intfloat/e5-small-v2`; the deterministic
   384-dimensional hashing vectorizer is only a CPU-safe fallback.
4. Use the same encoder for query vectors, learned dense retrieval, pgvector
   retrieval, and embedding-space analysis; no cloud embedding API is required.
5. Search through FastAPI endpoints:
   - `/search/bm25`: lexical baseline with query rewriting and identifier lookup
   - `/search/dense`: learned dense retrieval over pgvector/HNSW
   - `/search/pgvector`: explicit debug endpoint for the same ANN index path
   - `/search/hybrid`: multi-stage lexical + learned dense retrieval, weighted
     fusion, heuristic reranking, and optional LLM reranking (DeepSeek)

## Dataset

The current Docker-backed local dataset contains:

- `raw_news`: 518 768
- `raw_logs`: 9 718
- `structured_events`: 211
- total counted objects: 528 739

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

A local embedding client was added for deterministic fallback and learned
sentence-transformer retrieval over `structured_events.embedding`. Hybrid
retrieval fuses BM25 and learned dense rankings with reciprocal-rank fusion.

This improves vocabulary mismatch and repository/time cases while keeping BM25's
exact-match strengths for Statuspage and OSV identifiers.

### Iteration 3: pgvector Structured Events

Validation-relevant news records are transformed into `structured_events`,
embedded with the same local encoder used for query vectors, and searched
through PostgreSQL `pgvector`. This provides a real vector database path that is
fast on CPU and does not require external embedding API keys.

The current pgvector path has high recall but can return hard negatives close to
the positives. That is expected for same-source-type negatives and motivates a
future reranking stage.

### Iteration 4: Multi-Stage Reranking

The hybrid endpoint now runs a cascade:

1. exact and expanded BM25 retrieval;
2. pgvector HNSW retrieval over learned structured-event embeddings;
3. weighted fusion with provider/source/time heuristics;
4. optional LLM reranking (DeepSeek) over the top shortlist.

This follows the course pattern of using cheaper recall-oriented stages first
and applying heavier ranking only to a small candidate set.

### Iteration 5: Embedding-Space Analysis

`evaluation.embedding_analysis` measures query-vector norms, positive
similarity, hardest-negative similarity, margin, and separation rate on the
blind validation set. It uses the same query/document prefixing as production
retrieval, so the report reflects the deployed embedding space.

## Evaluation

Run:

```powershell
$env:DATABASE_URL='postgresql+asyncpg://postgres:postgres@127.0.0.1:55432/incident_news_search'
py -m evaluation.validate_retrieval --top-k 10
py -m evaluation.validate_qrels --top-k 10
py -m evaluation.embedding_analysis --backend auto
py -m evaluation.benchmark_real --benchmark-document-embeddings --embedding-sample-size 100
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
| BM25 | 0.56 | 0.50 | 0.52 | 0.00 |
| Dense | 1.00 | 0.95 | 0.96 | 0.58 |
| pgvector | 1.00 | 0.95 | 0.96 | 0.58 |
| Hybrid + LLM | 1.00 | 0.98 | 0.99 | 0.58 |

Current blind qrels summary:

| Mode | nDCG@10 | MRR@10 | Recall@10 | Precision@10 | Hard-negative hit@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| BM25 | 0.23 | 0.25 | 0.17 | 0.21 | 0.00 |
| Dense | 0.84 | 0.90 | 0.64 | 0.17 | 0.45 |
| pgvector | 0.84 | 0.90 | 0.64 | 0.17 | 0.45 |
| Hybrid + LLM | 0.84 | 0.89 | 0.66 | 0.17 | 0.46 |

This blind qrels result is the more honest offline estimate. The linked result
is useful as a pipeline sanity check but overstates semantic retrieval quality
because some queries contain strong source-specific identifiers.

Current real pgvector benchmark on 100 validation queries:

| Stage | mean ms | p50 ms | p95 ms |
| --- | ---: | ---: | ---: |
| Query embedding | 26.7 | 26.5 | 31.9 |
| HNSW database search | 2.5 | 2.4 | 2.8 |
| End-to-end pgvector | 29.2 | 28.9 | 34.4 |

The benchmark confirmed `HNSW index used=True` and `sequential scan used=False`.

Read-only document embedding benchmark:

| metric | value |
| --- | ---: |
| sample size | 100 |
| total generation time | 9.748 s |
| throughput | 10.26 docs/s |
| mean per document | 97.5 ms |
| p95 per document | 150.4 ms |

## Hardware Requirements

Development and demo inference have been tested on a local laptop with Docker
Desktop and PostgreSQL/pgvector. The included dump is about 103 MB compressed.
The database restores into a Docker volume and serves local demo traffic on CPU.

The current default retrieval path does not require GPU. Sentence-transformer
analysis can run on CPU; GPU is optional for faster neural embedding experiments.
LLM reranking (DeepSeek) is optional and only runs when explicitly enabled.

## Recommended Next Iteration

- Refresh all `structured_events` embeddings after changing `EMBEDDING_MODEL`.
- Store chunk-level embeddings for long news bodies.
- Add a cross-encoder or lightweight reranker for the top 20 hybrid candidates.
- Add query rewriting/HyDE for short logs that lack provider-specific terms.
