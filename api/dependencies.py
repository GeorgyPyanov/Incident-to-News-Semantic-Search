from __future__ import annotations

from collections.abc import Sequence

from api.pipeline import IncidentNewsSearchPipeline
from event_extraction.service import IncidentExtractionService
from retrieval.reasoning import NewsReasoningService
from retrieval.schemas import NewsArticle
from retrieval.service import InMemoryNewsRetriever


def build_search_pipeline(news_articles: Sequence[NewsArticle] | None = None) -> IncidentNewsSearchPipeline:
    return IncidentNewsSearchPipeline(
        extractor=IncidentExtractionService(),
        retriever=InMemoryNewsRetriever(news_articles),
        reasoner=NewsReasoningService(),
    )
