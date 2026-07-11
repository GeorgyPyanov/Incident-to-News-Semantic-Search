from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from retrieval.schemas import NewsArticle


DEFAULT_DATASET_PATH = Path(__file__).parent / "data" / "evaluation_dataset.json"


@dataclass(frozen=True)
class EvaluationQuery:
    id: str
    incident_log: str
    candidate_articles: tuple[NewsArticle, ...]
    relevant_article_ids: frozenset[str]


@dataclass(frozen=True)
class EvaluationDataset:
    queries: tuple[EvaluationQuery, ...]


def load_evaluation_dataset(path: str | Path = DEFAULT_DATASET_PATH) -> EvaluationDataset:
    dataset_path = Path(path)
    with dataset_path.open("r", encoding="utf-8") as dataset_file:
        raw_dataset = json.load(dataset_file)

    queries = tuple(_parse_query(raw_query) for raw_query in raw_dataset["queries"])
    return EvaluationDataset(queries=queries)


def _parse_query(raw_query: dict[str, object]) -> EvaluationQuery:
    raw_articles = raw_query.get("candidate_articles", [])
    if not isinstance(raw_articles, list):
        raise ValueError("candidate_articles must be a list")

    raw_relevant_ids = raw_query.get("relevant_article_ids", [])
    if not isinstance(raw_relevant_ids, list):
        raise ValueError("relevant_article_ids must be a list")

    return EvaluationQuery(
        id=str(raw_query["id"]),
        incident_log=str(raw_query.get("incident_log") or ""),
        candidate_articles=tuple(_parse_article(raw_article) for raw_article in raw_articles),
        relevant_article_ids=frozenset(str(article_id) for article_id in raw_relevant_ids),
    )


def _parse_article(raw_article: dict[str, object]) -> NewsArticle:
    return NewsArticle(
        id=str(raw_article["id"]),
        title=str(raw_article.get("title") or ""),
        url=str(raw_article.get("url") or ""),
        source=_optional_str(raw_article.get("source")),
        published_at=_optional_str(raw_article.get("published_at")),
        content=_optional_str(raw_article.get("content")),
    )


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None

