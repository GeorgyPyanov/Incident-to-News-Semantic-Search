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
