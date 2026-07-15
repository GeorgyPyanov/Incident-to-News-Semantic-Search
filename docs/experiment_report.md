# Experiment Report

## Project Goal

Incident-to-News Semantic Search helps an operator connect a short incident log
to relevant external evidence. The system retrieves statuspage incidents,
security advisories, GitHub releases, news stories, and event records, then can
produce a cited RAG answer when the retrieved evidence is strong enough.

The target workflow is investigation support: an SRE, support engineer, or
security analyst starts with a terse log or outage description and needs related
source material quickly.

Learning problems:

- represent heterogeneous incident/news records as comparable vectors;
- recover relevant evidence when the query and document use different wording;
- keep vector search fast on a local CPU-backed PostgreSQL setup;
- decide when retrieved evidence is strong enough for a generated answer.

## System Design

The retrieval stack has four layers:

1. Lexical retrieval with PostgreSQL full-text search and identifier lookup.
2. Dense retrieval over `raw_news.embedding` with PostgreSQL pgvector and HNSW.
3. Hybrid fusion with either reciprocal-rank fusion (`rrf`) or normalized-score
   summation (`normalized_sum`).
4. Heuristic scoring and optional DeepSeek reranking over the fused shortlist.

The answer stack is a RAG path:

1. Retrieve evidence with the configured search mode, `hybrid` by default.
2. Apply abstention gates for top score, supporting documents, and lexical
   overlap.
3. Send retrieved titles, snippets, URLs, and metadata to the configured LLM.
4. Validate JSON output and render a cited answer.

DeepSeek is used for LLM reranking and is the default answer generator. Ollama
is supported only for local answer generation through
`RAG_GENERATOR_PROVIDER=ollama`.

## Data Model

The local large dataset contains:

| table | rows | role |
| --- | ---: | --- |
| `raw_news` | 518,768 | primary retrieval corpus |
| `raw_logs` | 9,718 | incident/log inputs |
| `structured_events` | 211 | extracted diagnostics and validation links |

`raw_news.embedding` stores 384-dimensional vectors. The HNSW index
`ix_raw_news_embedding` supports the dense retrieval endpoint and the dense
stage inside hybrid search.

Source families:

- Statuspage incident APIs
- OSV.dev advisories
- GH Archive and GitHub releases
- Hacker News and Google News
- GDELT

## Embedding Model

The default learned encoder is `intfloat/e5-small-v2` because it produces
384-dimensional vectors, matches the `query:` and `passage:` prefix convention,
and runs locally on CPU. The deterministic `hashing-vectorizer-384` backend is
available for reproducible local benchmarking and fallback runs.

The benchmark suite also supports CPU dynamic quantization for the
SentenceTransformer encoder:

```powershell
$env:EMBEDDING_QUANTIZATION='dynamic'
```

The comparison runner evaluates embedding and fusion variants while keeping the
same pgvector schema:

```powershell
py -m evaluation.compare_iterations --top-k 10 --embedding-sample-size 100
```

## Retrieval Details

For `top_k=10`, the hybrid path uses:

| stage | candidate count |
| --- | ---: |
| original BM25 query | 120 |
| rewritten BM25 query | 120 |
| dense pgvector query | 120 |
| fused unique candidates | up to 360 |
| heuristic scoring | all fused candidates |
| DeepSeek shortlist | 20 before provider cap |
| DeepSeek payload | 12 by `DEEPSEEK_RERANK_TOP_N` |
| final response | 10 |

The heuristic ranker rewards lexical overlap, exact identifiers, provider
matches, source-type priors, and date proximity. The final hybrid score is a
weighted combination of fusion, heuristic, and optional LLM scores.

## RAG Answering

`POST /answer` returns one of three statuses:

| status | meaning |
| --- | --- |
| `answered` | evidence passed the gates and the LLM returned a valid cited answer |
| `abstained` | retrieved evidence was too weak or the answer failed validation |
| `llm_unavailable` | evidence was present, but the configured LLM call failed |

The generator is instructed to use only retrieved documents. The service
validates that each answer section cites supplied documents and retries once
with repair feedback when the JSON draft is malformed or under-cited.

## Validation Sets

The evaluation data contains 150 labeled examples:

- 50 Statuspage logs linked by incident identifier
- 50 OSV logs linked by advisory identifier
- 50 GitHub activity logs linked by repository and time window

Two validation views are used:

- linked validation: source-linked sanity checks
- blind qrels validation: direct identifiers and title-like query hints removed

The blind qrels use graded relevance:

| grade | meaning |
| ---: | --- |
| 3 | same incident, advisory, or release |
| 1 | topically related weaker match |
| 0 | unrelated or different linkage |

## Experimental Pipeline

Each experiment uses the same evaluation flow:

1. prepare or refresh `raw_news` embeddings;
2. run linked validation and blind qrels validation at `top_k=10`;
3. measure pgvector/HNSW latency with `evaluation.benchmark_real`;
4. analyze vector separation, margins, PCA, t-SNE, and effective rank;
5. compare BM25, dense pgvector, hybrid fusion, and hybrid + DeepSeek rerank.

The optimization knobs are embedding backend/model, dynamic quantization,
fusion mode (`rrf` or `normalized_sum`), and optional DeepSeek reranking.

## Quality Results

### Blind Qrels

Full run on 150 blind queries with DeepSeek reranking enabled for hybrid search.

| mode | nDCG@10 | MRR@10 | Recall@10 | Precision@10 | mean ms | p95 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25 | 0.234 | 0.246 | 0.172 | 0.209 | 69.5 | 115.2 |
| Dense | 0.248 | 0.271 | 0.163 | 0.030 | 54.8 | 91.5 |
| pgvector | 0.248 | 0.271 | 0.163 | 0.030 | 37.9 | 49.4 |
| Hybrid + DeepSeek | 0.345 | 0.376 | 0.226 | 0.040 | 7210.1 | 9511.4 |

Blind qrels are the stricter offline estimate because query text has fewer
source-specific shortcuts.

### Linked Validation

Linked validation checks whether the system can recover known source-linked
positives.

| mode | hit@10 | nDCG@10 | MRR@10 | Recall@10 | mean ms | p95 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25 | 0.560 | 0.515 | 0.500 | 0.560 | 399.0 | 1248.0 |
| Dense | 0.367 | 0.344 | 0.337 | 0.367 | 40.0 | 116.1 |
| pgvector | 0.367 | 0.344 | 0.337 | 0.367 | 16.6 | 28.9 |
| Hybrid + DeepSeek | 0.760 | 0.724 | 0.712 | 0.760 | 8258.4 | 11053.2 |

The linked set is useful as a pipeline sanity check. It is easier than blind
qrels because many examples preserve strong source-specific identifiers.

## Comparative Conclusion

The most efficient low-latency retrieval path is `pgvector`: it matches dense
quality on the saved blind qrels run while reducing mean latency from 54.8 ms
to 37.9 ms. The best quality path is `Hybrid + DeepSeek`: it improves blind
`nDCG@10` from 0.248 to 0.345 and linked `hit@10` from 0.367 to 0.760, but the
LLM rerank stage raises latency to seconds.

For interactive search, pgvector/hybrid without LLM rerank is the practical
default. For offline investigation or high-value queries, DeepSeek reranking is
worth the latency because it gives the strongest ranking quality.

## Vector Analysis

Embedding-space analysis on the blind validation set:

| metric | value |
| --- | ---: |
| separation rate | 0.900 |
| mean positive cosine | 0.439 |
| mean nearest negative cosine | 0.180 |
| mean margin | 0.258 |
| effective rank | 174.0 |

`evaluation.embedding_analysis` also writes PCA coordinates, t-SNE coordinates,
per-dimension variance statistics, and effective rank to
`evaluation/embedding_analysis.json`.

## System Benchmark

Read-only benchmark over stored `raw_news` embeddings:

| metric | value |
| --- | ---: |
| embedded raw news | 518,768 / 518,768 |
| embedding coverage | 100.0% |
| embedding model | `hashing-vectorizer-384` |
| HNSW index size | 994 MiB |
| query embedding mean / p95 | 0.265 ms / 0.333 ms |
| database index search mean / p95 | 56.71 ms / 131.11 ms |
| end-to-end retrieval mean / p95 | 56.98 ms / 131.35 ms |

Read-only document embedding benchmark:

| metric | value |
| --- | ---: |
| sample size | 100 |
| total generation time | 0.0112 s |
| throughput | 8,932.32 docs/s |
| mean per document | 0.112 ms |
| p95 per document | 0.134 ms |

The benchmark records whether the HNSW index is used and whether PostgreSQL
falls back to a sequential scan. The saved benchmark reports HNSW usage and no
sequential scan for the measured pgvector query.

## Reproducibility

Main commands:

```powershell
$env:DATABASE_URL='postgresql+asyncpg://postgres:postgres@127.0.0.1:55432/incident_news_search'
py -m data.embed_raw_news --all
py -m evaluation.validate_retrieval --top-k 10
py -m evaluation.validate_qrels --top-k 10
py -m evaluation.embedding_analysis --backend auto
py -m evaluation.benchmark_real --benchmark-document-embeddings --embedding-sample-size 100
py -m evaluation.compare_iterations --top-k 10 --embedding-sample-size 100
```

Runtime switches:

```powershell
$env:EMBEDDING_BACKEND='auto'
$env:EMBEDDING_MODEL='intfloat/e5-small-v2'
$env:EMBEDDING_QUANTIZATION='none'
$env:RETRIEVAL_FUSION_MODE='rrf'
$env:DEEPSEEK_RERANK_ENABLED='true'
$env:DEEPSEEK_RERANK_TOP_N='12'
$env:RAG_GENERATOR_PROVIDER='deepseek'
```

## Hardware

The recorded local benchmark ran on:

| item | value |
| --- | --- |
| OS | Windows 10 |
| Python | 3.11.2 |
| CPU | AMD Ryzen 5 5600H with Radeon Graphics |
| cores | 12 logical / 6 physical |
| RAM | 15.36 GiB |
| GPU | No GPU |

The default retrieval path runs on CPU. GPU is optional for faster embedding
experiments, not required for the API demo.

## Limitations

- GitHub release relevance is often repository/time based, which remains harder
  than exact identifier matching.
- Long news bodies are represented as single document vectors; chunk-level
  embeddings would improve long-document evidence retrieval.
- DeepSeek reranking improves ranking quality but adds seconds of latency.
- RAG answers are intentionally conservative and abstain when retrieved
  evidence is weak.
