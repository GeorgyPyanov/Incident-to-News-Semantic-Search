"""Embedding generation utilities backed by the OpenAI API."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - exercised when the optional dependency is absent.
    OpenAI = None  # type: ignore[assignment]


class EmbeddingClient(Protocol):
    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        raise NotImplementedError


@dataclass(slots=True)
class OpenAIEmbeddingClient:
    model: str = "text-embedding-3-small"
    api_key: str | None = None
    _client: object = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        # Keep the OpenAI client lazy and isolated behind a tiny adapter.
        if OpenAI is None:
            raise RuntimeError("The 'openai' package is required to use OpenAIEmbeddingClient.")
        self._client = OpenAI(api_key=self.api_key)

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        cleaned = [text.strip() for text in texts]
        response = self._client.embeddings.create(model=self.model, input=cleaned)
        vectors = [item.embedding for item in response.data]
        if len(vectors) != len(cleaned):
            raise RuntimeError("OpenAI returned a mismatched number of embeddings.")
        return vectors

    def embed_text(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]
