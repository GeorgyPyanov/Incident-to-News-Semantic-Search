from __future__ import annotations

from typing import Protocol

from api.schemas import IncidentSearchRequest, IncidentSearchResponse, NewsResultResponse
from event_extraction.schemas import IncidentData
from event_extraction.service import IncidentExtractionService
from retrieval.reasoning import NewsReasoningService
from retrieval.schemas import RetrievedNewsResult
from retrieval.service import InMemoryNewsRetriever


class NewsRetriever(Protocol):
    def search(self, incident: IncidentData | str, top_k: int = 5) -> list[RetrievedNewsResult]:
        ...


class IncidentNewsSearchPipeline:
    def __init__(
        self,
        extractor: IncidentExtractionService | None = None,
        retriever: NewsRetriever | None = None,
        reasoner: NewsReasoningService | None = None,
    ) -> None:
        self._extractor = extractor or IncidentExtractionService()
        self._retriever = retriever or InMemoryNewsRetriever()
        self._reasoner = reasoner or NewsReasoningService()

    def search(self, request: IncidentSearchRequest) -> IncidentSearchResponse:
        incident = request.incident or self._extractor.extract(request.original_log)
        retrieved_results = self._retriever.search(incident, top_k=request.top_k)

        results = [
            NewsResultResponse(
                id=result.article.id,
                title=result.article.title,
                url=result.article.url,
                source=result.article.source,
                published_at=result.article.published_at,
                score=result.score,
                reasoning=self._reasoner.explain(incident, result.article),
            )
            for result in retrieved_results
        ]
        return IncidentSearchResponse(results=results)
