"""Open-source embedding utilities."""

from __future__ import annotations

import hashlib
import math
import os
import re
from dataclasses import dataclass, field
from typing import Literal, Protocol, Sequence

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


EmbeddingQuantization = Literal["none", "dynamic"]


@dataclass(slots=True)
class SentenceTransformerEmbeddingClient:
    model: str = "intfloat/e5-small-v2"
    device: str | None = None
    normalize_embeddings: bool = True
    quantization: EmbeddingQuantization = "none"
    _client: object = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if SentenceTransformer is None:
            raise RuntimeError(
                "The 'sentence-transformers' package is required for SentenceTransformerEmbeddingClient."
            )
        self._client = SentenceTransformer(self.model, device=self.device)
        self.quantization = _normalize_quantization(self.quantization)
        if self.quantization == "dynamic":
            if self.device not in (None, "cpu"):
                raise ValueError("Dynamic quantization is only supported on CPU.")
            _apply_dynamic_quantization(self._client)

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
        if self.quantization == "none":
            return self.model
        return f"{self.model}@{self.quantization}"


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
    quantization: EmbeddingQuantization = "none",
) -> EmbeddingClient:
    backend_name = backend.strip().lower()
    if backend_name == "hashing":
        return HashingEmbeddingClient(dimensions=dimensions)
    if backend_name in {"sentence-transformer", "sentence_transformer", "sentence-transformers", "auto"}:
        try:
            return SentenceTransformerEmbeddingClient(
                model=model,
                device=device,
                quantization=quantization,
            )
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


def _normalize_quantization(value: EmbeddingQuantization | str) -> EmbeddingQuantization:
    normalized = str(value or "none").strip().lower()
    if normalized not in {"none", "dynamic"}:
        raise ValueError(f"Unsupported embedding quantization mode: {value!r}")
    return normalized  # type: ignore[return-value]


def _apply_dynamic_quantization(client: object) -> None:
    try:
        import torch
    except ImportError as error:  # pragma: no cover - dependency error path
        raise RuntimeError("Dynamic quantization requires torch.") from error

    transformer_module = _first_transformer_module(client)
    if transformer_module is None:
        raise RuntimeError("SentenceTransformer model does not expose a quantizable transformer module.")

    model = getattr(transformer_module, "auto_model", None) or getattr(transformer_module, "model", None)
    if model is None:
        raise RuntimeError("SentenceTransformer transformer module does not expose an underlying model.")

    quantize_dynamic = getattr(getattr(getattr(torch, "ao", None), "quantization", None), "quantize_dynamic", None)
    if quantize_dynamic is None:
        quantize_dynamic = torch.quantization.quantize_dynamic

    quantized_model = quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    if hasattr(transformer_module, "auto_model"):
        transformer_module.auto_model = quantized_model
    else:
        transformer_module.model = quantized_model


def _first_transformer_module(client: object) -> object | None:
    modules = getattr(client, "_modules", None)
    if isinstance(modules, dict) and modules:
        return next(iter(modules.values()))
    return None


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
