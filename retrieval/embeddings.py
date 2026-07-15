"""Open-source embedding utilities."""

from __future__ import annotations

import hashlib
import math
import os
import re
from dataclasses import dataclass, field
from typing import Protocol, Sequence

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover - optional dependency
    SentenceTransformer = None  # type: ignore[assignment]


class EmbeddingClient(Protocol):
    model_name: str

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        raise NotImplementedError

    def embed_text(self, text: str) -> list[float]:
        raise NotImplementedError


@dataclass(slots=True)
class SentenceTransformerEmbeddingClient:
    model: str = "intfloat/e5-small-v2"
    device: str | None = None
    normalize_embeddings: bool = True
    _client: object = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if SentenceTransformer is None:
            raise RuntimeError(
                "The 'sentence-transformers' package is required for SentenceTransformerEmbeddingClient."
            )
        self._client = SentenceTransformer(self.model, device=self.device)

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        cleaned = [text.strip() for text in texts]
        vectors = self._client.encode(
            cleaned,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=False,
        )
        return [vector.tolist() if hasattr(vector, "tolist") else list(vector) for vector in vectors]

    def embed_text(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]

    @property
    def model_name(self) -> str:
        return self.model


@dataclass(slots=True)
class HashingEmbeddingClient:
    dimensions: int = 384
    normalize_embeddings: bool = True

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        return [_hashed_vector(text, self.dimensions, self.normalize_embeddings) for text in texts]

    def embed_text(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]

    @property
    def model_name(self) -> str:
        return f"hashing-vectorizer-{self.dimensions}"


def build_embedding_client(
    *,
    backend: str = "auto",
    model: str = "intfloat/e5-small-v2",
    dimensions: int = 384,
    device: str | None = None,
) -> EmbeddingClient:
    backend_name = backend.strip().lower()
    if backend_name == "hashing":
        return HashingEmbeddingClient(dimensions=dimensions)
    if backend_name in {"sentence-transformer", "sentence_transformer", "sentence-transformers", "auto"}:
        try:
            return SentenceTransformerEmbeddingClient(model=model, device=device)
        except Exception:
            if backend_name != "auto":
                raise
            return HashingEmbeddingClient(dimensions=dimensions)
    raise ValueError(f"Unknown embedding backend: {backend}")


def build_query_text(text: str) -> str:
    return _with_prefix(text, os.environ.get("EMBEDDING_QUERY_PREFIX", "query: "))


def build_document_text(text: str) -> str:
    return _with_prefix(text, os.environ.get("EMBEDDING_DOCUMENT_PREFIX", "passage: "))


def validate_embedding_dimension(vector: Sequence[float], expected_dimensions: int, model_name: str) -> None:
    actual_dimensions = len(vector)
    if actual_dimensions != expected_dimensions:
        raise ValueError(
            f"Embedding model {model_name!r} produced {actual_dimensions} dimensions; "
            f"database expects {expected_dimensions}. Update EMBEDDING_DIM or use a matching model."
        )


def _with_prefix(text: str, prefix: str | None) -> str:
    normalized = (text or "").strip()
    prefix_text = (prefix or "").strip()
    if not prefix_text:
        return normalized
    lower_prefix = prefix_text.lower()
    if normalized.lower().startswith(lower_prefix):
        suffix = normalized[len(prefix_text):].lstrip()
        return f"{prefix_text} {suffix}".rstrip() if suffix else prefix_text
    return f"{prefix_text} {normalized}"


TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_-]{2,}", re.IGNORECASE)


def _hashed_vector(text: str, dimensions: int, normalize: bool) -> list[float]:
    values = [0.0] * dimensions
    for token in TOKEN_RE.findall(text or ""):
        digest = hashlib.blake2b(token.lower().encode("utf-8"), digest_size=8).digest()
        bucket = int.from_bytes(digest[:4], "big") % dimensions
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        values[bucket] += sign
    if not normalize:
        return values
    norm = math.sqrt(sum(value * value for value in values))
    if norm:
        values = [value / norm for value in values]
    return values
