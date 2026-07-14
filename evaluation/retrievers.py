from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from typing import Protocol

from retrieval.schemas import NewsArticle, RetrievedNewsResult
from retrieval.service import InMemoryNewsRetriever


TOKEN_PATTERN = re.compile(r"[a-z0-9][a-z0-9_-]{2,}", re.IGNORECASE)


class EvaluationRetriever(Protocol):
    name: str
    config: dict[str, object]

    def search(
        self,
        incident_log: str,
        candidate_articles: tuple[NewsArticle, ...],
        top_k: int,
    ) -> list[RetrievedNewsResult]:
        ...


@dataclass(frozen=True)
class KeywordRetriever:
    name: str = "keyword_lexical"
    min_score: float = 0.0

    @property
    def config(self) -> dict[str, object]:
        return {"scoring": "token_overlap_jaccard", "min_score": self.min_score}

    def search(
        self,
        incident_log: str,
        candidate_articles: tuple[NewsArticle, ...],
        top_k: int,
    ) -> list[RetrievedNewsResult]:
        query_tokens = _tokens(incident_log)
        results = [
            RetrievedNewsResult(article=article, score=_jaccard(query_tokens, _tokens(article_embedding_text(article))))
            for article in candidate_articles
        ]
        return _rank(results, top_k, self.min_score)


@dataclass(frozen=True)
class SemanticEmbeddingRetriever:
    dimensions: int = 32
    name: str = "semantic_embedding"
    min_score: float = -1.0

    @property
    def config(self) -> dict[str, object]:
        return {
            "embedding": "deterministic_hashing_token_average",
            "dimensions": self.dimensions,
            "min_score": self.min_score,
        }

    def search(
        self,
        incident_log: str,
        candidate_articles: tuple[NewsArticle, ...],
        top_k: int,
    ) -> list[RetrievedNewsResult]:
        query_vector = generate_text_embedding(incident_log, self.dimensions)
        results = [
            RetrievedNewsResult(
                article=article,
                score=cosine_similarity(
                    query_vector,
                    generate_text_embedding(article_embedding_text(article), self.dimensions),
                ),
            )
            for article in candidate_articles
        ]
        return _rank(results, top_k, self.min_score)


@dataclass(frozen=True)
class HybridRetriever:
    lexical_weight: float = 0.55
    semantic_weight: float = 0.45
    dimensions: int = 32
    name: str = "hybrid"
    min_score: float = -1.0

    @property
    def config(self) -> dict[str, object]:
        return {
            "lexical_weight": self.lexical_weight,
            "semantic_weight": self.semantic_weight,
            "semantic_dimensions": self.dimensions,
            "min_score": self.min_score,
        }

    def search(
        self,
        incident_log: str,
        candidate_articles: tuple[NewsArticle, ...],
        top_k: int,
    ) -> list[RetrievedNewsResult]:
        query_tokens = _tokens(incident_log)
        query_vector = generate_text_embedding(incident_log, self.dimensions)
        results: list[RetrievedNewsResult] = []
        for article in candidate_articles:
            article_text = article_embedding_text(article)
            lexical_score = _jaccard(query_tokens, _tokens(article_text))
            semantic_score = cosine_similarity(
                query_vector,
                generate_text_embedding(article_text, self.dimensions),
            )
            score = (self.lexical_weight * lexical_score) + (self.semantic_weight * semantic_score)
            results.append(RetrievedNewsResult(article=article, score=score))
        return _rank(results, top_k, self.min_score)


@dataclass(frozen=True)
class CurrentDefaultRetrieverAdapter:
    name: str = "current_default"

    @property
    def config(self) -> dict[str, object]:
        return {"implementation": "retrieval.service.InMemoryNewsRetriever"}

    def search(
        self,
        incident_log: str,
        candidate_articles: tuple[NewsArticle, ...],
        top_k: int,
    ) -> list[RetrievedNewsResult]:
        retriever = InMemoryNewsRetriever(candidate_articles)
        return retriever.search(incident_log, top_k=top_k)


def default_retrieval_approaches() -> tuple[EvaluationRetriever, ...]:
    return (
        KeywordRetriever(),
        SemanticEmbeddingRetriever(),
        HybridRetriever(),
        CurrentDefaultRetrieverAdapter(),
    )


def _tokens(text: str | None) -> set[str]:
    return {match.group(0).lower() for match in TOKEN_PATTERN.finditer(text or "")}


def article_embedding_text(article: NewsArticle) -> str:
    return " ".join(
        part
        for part in (
            article.title,
            article.source,
            article.published_at,
            article.content,
        )
        if part
    )


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def generate_text_embedding(text: str | None, dimensions: int) -> tuple[float, ...]:
    vector = [0.0] * dimensions
    tokens = sorted(_tokens(text))
    if not tokens:
        return tuple(vector)

    for token in tokens:
        token_vector = _token_embedding(token, dimensions)
        for index, value in enumerate(token_vector):
            vector[index] += value

    return _normalize(tuple(value / len(tokens) for value in vector))


def _token_embedding(token: str, dimensions: int) -> tuple[float, ...]:
    digest = hashlib.sha256(token.encode("utf-8")).digest()
    values = []
    for index in range(dimensions):
        byte = digest[index % len(digest)]
        values.append((byte / 127.5) - 1.0)
    return tuple(values)


def _normalize(vector: tuple[float, ...]) -> tuple[float, ...]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return tuple(value / norm for value in vector)


def cosine_similarity(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    if not left or not right:
        return 0.0
    return sum(left_value * right_value for left_value, right_value in zip(left, right))


def _rank(results: list[RetrievedNewsResult], top_k: int, min_score: float) -> list[RetrievedNewsResult]:
    filtered = [result for result in results if result.score >= min_score]
    filtered.sort(key=lambda result: (-result.score, result.article.id))
    return filtered[:top_k]
