from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Literal, Protocol, Sequence, cast

import httpx

from retrieval.db_search import DbNewsHit, SearchMode


class SearchBackend(Protocol):
    def search(self, query: str, mode: SearchMode, top_k: int = 10) -> list[DbNewsHit]: ...


class AnswerGenerator(Protocol):
    model: str

    def generate(self, query: str, hits: Sequence[DbNewsHit], *, repair_feedback: str | None = None) -> AnswerDraft: ...
    def close(self) -> None: ...


AnswerStatus = Literal["answered", "abstained", "llm_unavailable"]
TOKEN_RE = re.compile(r"[^\W_][^\W_0-9-]{2,}|[a-z0-9][a-z0-9_-]{2,}", re.IGNORECASE)
INLINE_CITATION_RE = re.compile(r"\[(\d+)\]")
STOPWORDS = {
    "как",
    "что",
    "это",
    "для",
    "при",
    "или",
    "the",
    "and",
    "for",
    "with",
    "from",
    "this",
    "that",
    "will",
    "may",
    "are",
    "our",
}

ABSTENTION_MESSAGE = "Я не знаю: найденные документы недостаточно релевантны для надежного ответа."
LLM_UNAVAILABLE_MESSAGE = "Релевантные документы найдены, но LLM сейчас недоступна."


@dataclass(frozen=True, slots=True)
class AnswerCitation:
    id: str
    title: str
    url: str | None
    source: str
    source_type: str
    published_at: str | None
    rank: int
    score: float


@dataclass(frozen=True, slots=True)
class GeneratedAnswer:
    status: AnswerStatus
    answer: str
    citations: tuple[AnswerCitation, ...]
    retrieval_mode: SearchMode
    model: str
    abstention_reason: str | None = None


@dataclass(frozen=True, slots=True)
class AnswerSection:
    label: str
    text: str
    citation_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class AnswerDraft:
    sections: tuple[AnswerSection, ...]


@dataclass(slots=True)
class AbstentionPolicy:
    min_top_score: float = field(default_factory=lambda: float(os.getenv("RAG_MIN_TOP_SCORE", "0.35")))
    min_supporting_documents: int = field(default_factory=lambda: int(os.getenv("RAG_MIN_SUPPORTING_DOCS", "1")))
    min_evidence_overlap: float = field(default_factory=lambda: float(os.getenv("RAG_MIN_EVIDENCE_OVERLAP", "0.12")))

    def reason(self, query: str, hits: Sequence[DbNewsHit]) -> str | None:
        if not hits:
            return "No relevant documents were retrieved."
        if hits[0].score < self.min_top_score:
            return (
                f"Top document score {hits[0].score:.3f} is below the "
                f"abstention threshold {self.min_top_score:.3f}."
            )
        supporting = sum(1 for hit in hits if hit.score >= self.min_top_score)
        if supporting < self.min_supporting_documents:
            return (
                f"Only {supporting} supporting documents met the relevance threshold; "
                f"{self.min_supporting_documents} required."
            )
        overlap = evidence_overlap(query, hits)
        if overlap < self.min_evidence_overlap:
            return (
                f"Evidence lexical overlap {overlap:.3f} is below the "
                f"abstention threshold {self.min_evidence_overlap:.3f}."
            )
        return None


@dataclass(slots=True)
class OllamaAnswerGenerator:
    model: str = field(default_factory=lambda: os.getenv("OLLAMA_MODEL", "qwen2.5:3b"))
    base_url: str = field(default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"))
    timeout_seconds: float = field(default_factory=lambda: float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "60")))
    _client: httpx.Client | None = field(init=False, default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        self._client = httpx.Client(base_url=self.base_url.rstrip("/"), timeout=self.timeout_seconds)

    def generate(self, query: str, hits: Sequence[DbNewsHit], *, repair_feedback: str | None = None) -> AnswerDraft:
        if self._client is None:
            raise RuntimeError("Ollama client is closed.")

        answer_requirements = [
            "Answer in Russian.",
            "Use only the supplied retrieved documents.",
            f'If the documents do not support a concrete answer, answer exactly: "{ABSTENTION_MESSAGE}".',
            "Do not invent facts, vendors, dates, fixes, commands, or root causes.",
            "Return strict JSON only, with this shape: {\"sections\":[{\"label\":\"...\",\"text\":\"...\",\"citation_ids\":[1]}]}.",
            "Return exactly three sections in this order: Короткий вывод, Что сделать, Почему это связано с источниками.",
            "Every section must include at least one citation id that refers to one of the supplied documents.",
            "Use only citation ids from the input documents.",
            "Keep the answer concise and actionable for an operator.",
            "Mention uncertainty when evidence is partial or only similar incidents are present.",
        ]
        if repair_feedback:
            answer_requirements.append(f"Repair this issue: {repair_feedback}")

        payload = {
            "model": self.model,
            "stream": False,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are an incident-response assistant. "
                        "Return JSON only. Do not include markdown or prose outside JSON."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "question": query,
                            "documents": [_document_payload(index, hit) for index, hit in enumerate(hits, start=1)],
                            "answer_requirements": answer_requirements,
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            "format": "json",
            "options": {"temperature": 0.0},
        }
        response = self._client.post("/api/chat", json=payload)
        response.raise_for_status()
        body = response.json()
        message = body.get("message") if isinstance(body, dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        if not content:
            raise RuntimeError("Ollama response did not contain message.content.")
        return _parse_draft(str(content).strip())

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None


@dataclass(slots=True)
class DeepSeekAnswerGenerator:
    model: str = field(default_factory=lambda: os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"))
    api_key: str | None = field(default_factory=lambda: os.getenv("DEEPSEEK_API_KEY"))
    base_url: str = field(default_factory=lambda: os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))
    timeout_seconds: float = field(default_factory=lambda: float(os.getenv("DEEPSEEK_TIMEOUT_SECONDS", "60")))
    max_tokens: int = field(default_factory=lambda: int(os.getenv("DEEPSEEK_ANSWER_MAX_TOKENS", "2500")))
    _client: httpx.Client | None = field(init=False, default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.api_key:
            self._client = None
            return
        self._client = httpx.Client(
            base_url=self.base_url.rstrip("/"),
            timeout=self.timeout_seconds,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
        )

    def generate(self, query: str, hits: Sequence[DbNewsHit], *, repair_feedback: str | None = None) -> AnswerDraft:
        if self._client is None:
            raise RuntimeError("DeepSeek API key is not configured.")

        answer_requirements = [
            "Answer in Russian.",
            "Use only the supplied retrieved documents.",
            f'If the documents do not support a concrete answer, answer exactly: "{ABSTENTION_MESSAGE}".',
            "Do not invent facts, vendors, dates, fixes, commands, or root causes.",
            "Return strict JSON only, with this shape: {\"sections\":[{\"label\":\"...\",\"text\":\"...\",\"citation_ids\":[1]}]}.",
            "Return exactly three sections in this order: РљРѕСЂРѕС‚РєРёР№ РІС‹РІРѕРґ, Р§С‚Рѕ СЃРґРµР»Р°С‚СЊ, РџРѕС‡РµРјСѓ СЌС‚Рѕ СЃРІСЏР·Р°РЅРѕ СЃ РёСЃС‚РѕС‡РЅРёРєР°РјРё.",
            "Every section must include at least one citation id that refers to one of the supplied documents.",
            "Use only citation ids from the input documents.",
            "Keep the answer concise and actionable for an operator.",
            "Mention uncertainty when evidence is partial or only similar incidents are present.",
        ]
        if repair_feedback:
            answer_requirements.append(f"Repair this issue: {repair_feedback}")

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are an incident-response assistant. "
                        "Return JSON only. Do not include markdown or prose outside JSON."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "question": query,
                            "documents": [_document_payload(index, hit) for index, hit in enumerate(hits, start=1)],
                            "answer_requirements": answer_requirements,
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            "temperature": 0,
            "max_tokens": self.max_tokens,
            "response_format": {"type": "json_object"},
        }
        response = self._client.post("/chat/completions", json=payload)
        response.raise_for_status()
        body = response.json()
        try:
            content = body["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as error:
            raise RuntimeError("DeepSeek response did not contain message.content.") from error
        if not content:
            raise RuntimeError("DeepSeek response did not contain message.content.")
        return _parse_draft(str(content).strip())

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None


@dataclass(slots=True)
class RagAnswerService:
    search_backend: SearchBackend
    generator: AnswerGenerator = field(default_factory=lambda: build_answer_generator())
    abstention_policy: AbstentionPolicy = field(default_factory=AbstentionPolicy)
    retrieval_mode: SearchMode = field(default_factory=lambda: _retrieval_mode(os.getenv("RAG_RETRIEVAL_MODE", "hybrid")))

    def answer(self, query: str, top_k: int = 5) -> GeneratedAnswer:
        top_k = max(1, min(top_k, 20))
        hits = self.search_backend.search(query, mode=self.retrieval_mode, top_k=top_k)
        citations = tuple(_citation(hit) for hit in hits)
        reason = self.abstention_policy.reason(query, hits)
        if reason is not None:
            return GeneratedAnswer(
                status="abstained",
                answer=ABSTENTION_MESSAGE,
                citations=citations,
                retrieval_mode=self.retrieval_mode,
                model=self.generator.model,
                abstention_reason=reason,
            )
        try:
            draft = self.generator.generate(query, hits)
            validation_reason = _validate_draft(draft, len(citations))
            if validation_reason is not None:
                draft = self.generator.generate(query, hits, repair_feedback=validation_reason)
                validation_reason = _validate_draft(draft, len(citations))
            if validation_reason is not None:
                return GeneratedAnswer(
                    status="abstained",
                    answer=ABSTENTION_MESSAGE,
                    citations=citations,
                    retrieval_mode=self.retrieval_mode,
                    model=self.generator.model,
                    abstention_reason=validation_reason,
                )
            answer = _render_answer(draft, citations)
            if ABSTENTION_MESSAGE in answer:
                return GeneratedAnswer(
                    status="abstained",
                    answer=ABSTENTION_MESSAGE,
                    citations=citations,
                    retrieval_mode=self.retrieval_mode,
                    model=self.generator.model,
                    abstention_reason="LLM abstained from the supplied evidence.",
                )
        except (httpx.HTTPError, RuntimeError) as error:
            return GeneratedAnswer(
                status="llm_unavailable",
                answer=LLM_UNAVAILABLE_MESSAGE,
                citations=citations,
                retrieval_mode=self.retrieval_mode,
                model=self.generator.model,
                abstention_reason=str(error),
            )
        return GeneratedAnswer(
            status="answered",
            answer=answer,
            citations=citations,
            retrieval_mode=self.retrieval_mode,
            model=self.generator.model,
        )


def build_answer_generator() -> AnswerGenerator:
    provider = str(os.getenv("RAG_GENERATOR_PROVIDER", "deepseek")).strip().lower()
    if provider in {"deepseek", "deepseek-chat"}:
        return DeepSeekAnswerGenerator()
    if provider in {"ollama", "local"}:
        return OllamaAnswerGenerator()
    raise ValueError(f"Unsupported RAG generator provider: {provider!r}")


def _citation(hit: DbNewsHit) -> AnswerCitation:
    return AnswerCitation(
        id=hit.id,
        title=hit.title,
        url=hit.url,
        source=hit.source,
        source_type=hit.source_type,
        published_at=hit.published_at,
        rank=hit.rank,
        score=hit.score,
    )


def _document_payload(index: int, hit: DbNewsHit) -> dict[str, object]:
    return {
        "citation": index,
        "id": hit.id,
        "title": hit.title,
        "url": hit.url,
        "source": hit.source,
        "source_type": hit.source_type,
        "published_at": hit.published_at,
        "score": hit.score,
        "snippet": hit.snippet,
    }


def evidence_overlap(query: str, hits: Sequence[DbNewsHit]) -> float:
    query_tokens = _tokens(query)
    if not query_tokens:
        return 0.0
    evidence_text = " ".join(
        part
        for hit in hits
        for part in (hit.title, hit.source, hit.source_type, hit.snippet or "")
        if part
    )
    evidence_tokens = _tokens(evidence_text)
    if not evidence_tokens:
        return 0.0
    return len(query_tokens & evidence_tokens) / len(query_tokens)


def _tokens(text: str) -> set[str]:
    return {
        match.group(0).lower()
        for match in TOKEN_RE.finditer(text or "")
        if match.group(0).lower() not in STOPWORDS
    }


def _parse_draft(content: str) -> AnswerDraft:
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as error:
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if not match:
            raise RuntimeError(f"LLM returned invalid JSON: {error}") from error
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError as nested_error:
            raise RuntimeError(f"LLM returned invalid JSON: {nested_error}") from nested_error

    if not isinstance(payload, dict):
        raise RuntimeError("LLM returned a non-object JSON payload.")
    sections = payload.get("sections")
    if not isinstance(sections, list):
        raise RuntimeError("LLM JSON payload does not contain a sections array.")

    parsed_sections: list[AnswerSection] = []
    for item in sections:
        if not isinstance(item, dict):
            raise RuntimeError("LLM JSON sections must be objects.")
        label = str(item.get("label") or "").strip()
        text = str(item.get("text") or "").strip()
        citation_ids_raw = item.get("citation_ids") or []
        if not label or not text:
            raise RuntimeError("Ollama JSON sections must include label and text.")
        if not isinstance(citation_ids_raw, list) or not citation_ids_raw:
            raise RuntimeError("Ollama JSON sections must include citation_ids.")
        citation_ids: list[int] = []
        for citation_id in citation_ids_raw:
            try:
                citation_int = int(citation_id)
            except (TypeError, ValueError) as error:
                raise RuntimeError("citation_ids must be integers.") from error
            citation_ids.append(citation_int)
        parsed_sections.append(AnswerSection(label=label, text=text, citation_ids=tuple(citation_ids)))
    return AnswerDraft(sections=tuple(parsed_sections))


def _validate_draft(draft: AnswerDraft, citations_count: int) -> str | None:
    if len(draft.sections) != 3:
        return "Answer must contain exactly three sections."
    expected_labels = (
        "короткий вывод",
        "что сделать",
        "почему это связано с источниками",
    )
    for section in draft.sections:
        if not section.text:
            return f"Section {section.label!r} is empty."
        if not section.citation_ids:
            return f"Section {section.label!r} is missing citation ids."
        if any(citation_id < 1 or citation_id > citations_count for citation_id in section.citation_ids):
            return f"Section {section.label!r} cites document indices outside the retrieved range of 1..{citations_count}."
    return None


def _render_answer(draft: AnswerDraft, citations: Sequence[AnswerCitation]) -> str:
    citation_map = {index: citation for index, citation in enumerate(citations, start=1)}
    rendered_lines: list[str] = []
    for index, section in enumerate(draft.sections, start=1):
        refs = " ".join(f"[{citation_id}]" for citation_id in section.citation_ids)
        rendered_lines.append(f"{index}. {section.label}: {section.text} {refs}".strip())
    rendered_lines.append("")
    rendered_lines.append("Источники:")
    for citation_id, citation in citation_map.items():
        source_line = f"[{citation_id}] {citation.title}"
        if citation.url:
            source_line += f" — {citation.url}"
        rendered_lines.append(source_line)
    return "\n".join(rendered_lines).strip()


def _retrieval_mode(value: str) -> SearchMode:
    normalized = str(value or "hybrid").strip().lower()
    if normalized not in {"bm25", "dense", "hybrid", "pgvector"}:
        raise ValueError(f"Unsupported RAG retrieval mode: {value!r}")
    return cast(SearchMode, normalized)
