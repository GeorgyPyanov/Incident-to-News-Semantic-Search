from collections.abc import Sequence
from uuid import UUID

from sqlalchemy import bindparam, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.engine import RowMapping
from sqlalchemy.ext.asyncio import AsyncSession

from database.schemas import (
    IncidentCreate,
    NewsSourceCreate,
    RawNewsCreate,
    ReasoningResultCreate,
    RetrievalLogCreate,
    StructuredEventCreate,
)
from database.table import Incident, NewsSource, RawNews, ReasoningResult, RetrievalLog, StructuredEvent


def _vector_literal(embedding: list[float]) -> str:
    return "[" + ",".join(str(float(value)) for value in embedding) + "]"


class NewsSourceDAL:
    def __init__(self, db_session: AsyncSession):
        self.db_session = db_session

    async def create(self, data: NewsSourceCreate) -> NewsSource:
        payload = data.model_dump()
        payload["url"] = str(payload["url"])
        payload["source_metadata"] = payload.pop("metadata")
        source = NewsSource(**payload)
        self.db_session.add(source)
        await self.db_session.flush()
        return source

    async def upsert(self, data: NewsSourceCreate) -> RowMapping:
        payload = data.model_dump()
        payload["url"] = str(payload["url"])
        payload["source_metadata"] = payload.pop("metadata")
        query = text("""
            INSERT INTO news_sources (
                name,
                source_type,
                url,
                country,
                region,
                city,
                language,
                enabled,
                poll_interval_seconds,
                metadata
            )
            VALUES (
                :name,
                :source_type,
                :url,
                :country,
                :region,
                :city,
                :language,
                :enabled,
                :poll_interval_seconds,
                :source_metadata
            )
            ON CONFLICT (name)
            DO UPDATE SET
                url = EXCLUDED.url,
                country = EXCLUDED.country,
                region = EXCLUDED.region,
                city = EXCLUDED.city,
                language = EXCLUDED.language,
                enabled = EXCLUDED.enabled,
                poll_interval_seconds = EXCLUDED.poll_interval_seconds,
                metadata = EXCLUDED.metadata,
                updated_at = now()
            RETURNING *
        """).bindparams(bindparam("source_metadata", type_=JSONB))
        result = await self.db_session.execute(query, payload)
        return result.mappings().one()

    async def list_enabled(self, source_type: str = "rss") -> Sequence[RowMapping]:
        result = await self.db_session.execute(
            text("""
                SELECT *
                FROM news_sources
                WHERE enabled = true
                  AND source_type = :source_type
                ORDER BY country, region NULLS LAST, city NULLS LAST, name
            """),
            {"source_type": source_type},
        )
        return result.mappings().all()

    async def mark_fetched(self, source_id: UUID) -> None:
        await self.db_session.execute(
            text("""
                UPDATE news_sources
                SET last_fetched_at = now(),
                    updated_at = now()
                WHERE id = :source_id
            """),
            {"source_id": source_id},
        )


class RawNewsDAL:
    def __init__(self, db_session: AsyncSession):
        self.db_session = db_session

    async def create(self, data: RawNewsCreate) -> RawNews:
        payload = data.model_dump()
        if payload["url"] is not None:
            payload["url"] = str(payload["url"])
        if payload["fetched_at"] is None:
            payload.pop("fetched_at")
        if payload["last_seen_at"] is None:
            payload.pop("last_seen_at")
        news = RawNews(**payload)
        self.db_session.add(news)
        await self.db_session.flush()
        return news

    async def create_or_touch(self, data: RawNewsCreate) -> RowMapping:
        payload = data.model_dump()
        if payload["url"] is not None:
            payload["url"] = str(payload["url"])

        query = text("""
                INSERT INTO raw_news (
                    source,
                    source_id,
                    source_type,
                    url,
                    url_hash,
                    title,
                    body,
                    language,
                    published_at,
                    fetched_at,
                    last_seen_at,
                    raw_region_hint,
                    raw_payload,
                    content_hash
                )
                VALUES (
                    :source,
                    :source_id,
                    :source_type,
                    :url,
                    :url_hash,
                    :title,
                    :body,
                    :language,
                    :published_at,
                    COALESCE(:fetched_at, now()),
                    COALESCE(:last_seen_at, now()),
                    :raw_region_hint,
                    :raw_payload,
                    :content_hash
                )
                ON CONFLICT (source, url_hash)
                WHERE url_hash IS NOT NULL
                DO UPDATE SET
                    last_seen_at = now(),
                    fetch_count = raw_news.fetch_count + 1,
                    raw_payload = EXCLUDED.raw_payload
                RETURNING *
            """).bindparams(bindparam("raw_payload", type_=JSONB))
        result = await self.db_session.execute(
            query,
            payload,
        )
        return result.mappings().one()

    async def get_by_url_hash(self, source: str, url_hash: str) -> RowMapping | None:
        result = await self.db_session.execute(
            text("""
                SELECT *
                FROM raw_news
                WHERE source = :source AND url_hash = :url_hash
                LIMIT 1
            """),
            {"source": source, "url_hash": url_hash},
        )
        return result.mappings().first()

    async def mark_duplicate(self, news_id: UUID, duplicate_of_id: UUID | None = None) -> None:
        await self.db_session.execute(
            text("""
                UPDATE raw_news
                SET is_duplicate = true,
                    duplicate_of_id = :duplicate_of_id,
                    processing_status = 'duplicate',
                    processed_at = now()
                WHERE id = :news_id
            """),
            {"news_id": news_id, "duplicate_of_id": duplicate_of_id},
        )

    async def mark_processed(self, news_id: UUID) -> None:
        await self.db_session.execute(
            text("""
                UPDATE raw_news
                SET processing_status = 'processed',
                    processed_at = now(),
                    extraction_error = NULL
                WHERE id = :news_id
            """),
            {"news_id": news_id},
        )

    async def mark_failed(self, news_id: UUID, error: str) -> None:
        await self.db_session.execute(
            text("""
                UPDATE raw_news
                SET processing_status = 'failed',
                    extraction_error = :error
                WHERE id = :news_id
            """),
            {"news_id": news_id, "error": error[:4000]},
        )

    async def list_unprocessed(self, limit: int = 100) -> Sequence[RowMapping]:
        result = await self.db_session.execute(
            text("""
                SELECT rn.*
                FROM raw_news rn
                WHERE rn.is_duplicate = false
                  AND rn.processed_at IS NULL
                  AND rn.processing_status IN ('new', 'failed')
                ORDER BY rn.published_at DESC NULLS LAST, rn.fetched_at DESC
                LIMIT :limit
            """),
            {"limit": limit},
        )
        return result.mappings().all()


class StructuredEventDAL:
    def __init__(self, db_session: AsyncSession):
        self.db_session = db_session

    async def create(self, data: StructuredEventCreate) -> StructuredEvent:
        payload = data.model_dump()
        payload["event_metadata"] = payload.pop("metadata")
        event = StructuredEvent(**payload)
        self.db_session.add(event)
        await self.db_session.flush()
        return event

    async def update_embedding(
        self,
        event_id: UUID,
        embedding: list[float],
        embedding_model: str,
    ) -> None:
        await self.db_session.execute(
            text("""
                UPDATE structured_events
                SET embedding = CAST(:embedding AS vector),
                    embedding_model = :embedding_model,
                    updated_at = now()
                WHERE id = :event_id
            """),
            {"event_id": event_id, "embedding": _vector_literal(embedding), "embedding_model": embedding_model},
        )

    async def search_by_embedding(
        self,
        embedding: list[float],
        limit: int = 100,
        regions: list[str] | None = None,
        published_after=None,
        published_before=None,
    ) -> Sequence[RowMapping]:
        conditions = ["embedding IS NOT NULL"]
        params = {
            "embedding": _vector_literal(embedding),
            "limit": limit,
            "regions": regions,
            "published_after": published_after,
            "published_before": published_before,
        }

        if regions:
            conditions.append("regions && :regions")
        if published_after is not None:
            conditions.append("published_at >= :published_after")
        if published_before is not None:
            conditions.append("published_at <= :published_before")

        result = await self.db_session.execute(
            text(f"""
                SELECT
                    *,
                    1 - (embedding <=> CAST(:embedding AS vector)) AS embedding_similarity
                FROM structured_events
                WHERE {" AND ".join(conditions)}
                ORDER BY embedding <=> CAST(:embedding AS vector)
                LIMIT :limit
            """),
            params,
        )
        return result.mappings().all()


class IncidentDAL:
    def __init__(self, db_session: AsyncSession):
        self.db_session = db_session

    async def create(self, data: IncidentCreate) -> Incident:
        incident = Incident(**data.model_dump())
        self.db_session.add(incident)
        await self.db_session.flush()
        return incident

    async def get_by_external_id(self, external_incident_id: str) -> RowMapping | None:
        result = await self.db_session.execute(
            text("""
                SELECT *
                FROM incidents
                WHERE external_incident_id = :external_incident_id
                LIMIT 1
            """),
            {"external_incident_id": external_incident_id},
        )
        return result.mappings().first()


class RetrievalLogDAL:
    def __init__(self, db_session: AsyncSession):
        self.db_session = db_session

    async def create_many(self, logs: Sequence[RetrievalLogCreate]) -> list[RetrievalLog]:
        rows = [RetrievalLog(**item.model_dump()) for item in logs]
        self.db_session.add_all(rows)
        await self.db_session.flush()
        return rows

    async def get_for_incident(self, incident_id: UUID) -> Sequence[RowMapping]:
        result = await self.db_session.execute(
            text("""
                SELECT rl.*, se.title, se.summary, se.event_type, se.provider, se.regions
                FROM retrieval_logs rl
                INNER JOIN structured_events se ON se.id = rl.event_id
                WHERE rl.incident_id = :incident_id
                ORDER BY rl.retrieval_stage, rl.rank_position
            """),
            {"incident_id": incident_id},
        )
        return result.mappings().all()


class ReasoningResultDAL:
    def __init__(self, db_session: AsyncSession):
        self.db_session = db_session

    async def create(self, data: ReasoningResultCreate) -> ReasoningResult:
        result = ReasoningResult(**data.model_dump())
        self.db_session.add(result)
        await self.db_session.flush()
        return result

    async def latest_for_incident(self, incident_id: UUID) -> RowMapping | None:
        result = await self.db_session.execute(
            text("""
                SELECT *
                FROM reasoning_results
                WHERE incident_id = :incident_id
                ORDER BY created_at DESC
                LIMIT 1
            """),
            {"incident_id": incident_id},
        )
        return result.mappings().first()
