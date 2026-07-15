from __future__ import annotations

import argparse
import json
import os
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from database.settings import settings
from retrieval.embeddings import build_document_text, build_embedding_client, build_query_text


DEFAULT_VALIDATION_SET = Path("evaluation/data/validation_blind.json")
DEFAULT_OUTPUT = Path("evaluation/embedding_analysis.json")


@dataclass(frozen=True)
class ExampleStats:
    query_norm: float
    positive_similarity: float
    hardest_negative_similarity: float

    @property
    def margin(self) -> float:
        return self.positive_similarity - self.hardest_negative_similarity

    @property
    def correctly_separated(self) -> bool:
        return self.positive_similarity > self.hardest_negative_similarity


def analyze(
    validation_path: Path,
    *,
    backend: str = "auto",
    model: str = "intfloat/e5-small-v2",
) -> dict[str, Any]:
    payload = json.loads(validation_path.read_text(encoding="utf-8"))
    examples = payload["examples"]
    client = build_embedding_client(backend=backend, model=model, dimensions=settings.embedding_dim)

    stats: list[ExampleStats] = []
    for example in examples:
        query_text = build_query_text(str(example["query"]["message"]))
        positive_text = build_document_text(_news_text(example["relevant_news"][0]))
        negative_texts = [build_document_text(_news_text(item)) for item in example.get("negative_news", [])]
        query_vector = client.embed_text(query_text)
        positive_vector = client.embed_text(positive_text)
        negative_vectors = [client.embed_text(text) for text in negative_texts] or [positive_vector]

        stats.append(
            ExampleStats(
                query_norm=_norm(query_vector),
                positive_similarity=_cosine(query_vector, positive_vector),
                hardest_negative_similarity=max(_cosine(query_vector, vector) for vector in negative_vectors),
            )
        )

    margins = [item.margin for item in stats]
    positive_sims = [item.positive_similarity for item in stats]
    negative_sims = [item.hardest_negative_similarity for item in stats]
    norms = [item.query_norm for item in stats]
    separated = sum(1 for item in stats if item.correctly_separated)

    return {
        "validation_set": str(validation_path),
        "backend": backend,
        "model": model,
        "examples": len(stats),
        "query_embedding_norm": _summary(norms),
        "positive_similarity": _summary(positive_sims),
        "hardest_negative_similarity": _summary(negative_sims),
        "margin": _summary(margins),
        "separation_rate": separated / len(stats) if stats else 0.0,
    }


def _news_text(item: dict[str, Any]) -> str:
    return " ".join(
        str(value).strip()
        for value in (
            item.get("title"),
            item.get("source"),
            item.get("url"),
            item.get("published_at"),
        )
        if value
    )


def _norm(vector: list[float]) -> float:
    return sum(value * value for value in vector) ** 0.5


def _cosine(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = _norm(left)
    right_norm = _norm(right)
    if not left_norm or not right_norm:
        return 0.0
    return dot / (left_norm * right_norm)


def _summary(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "mean": statistics.fmean(ordered) if ordered else 0.0,
        "median": statistics.median(ordered) if ordered else 0.0,
        "p95": ordered[min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))] if ordered else 0.0,
        "min": ordered[0] if ordered else 0.0,
        "max": ordered[-1] if ordered else 0.0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze the embedding space on the validation set.")
    parser.add_argument("--validation-set", type=Path, default=DEFAULT_VALIDATION_SET)
    parser.add_argument("--backend", default="auto")
    parser.add_argument("--model", default=os.getenv("EMBEDDING_MODEL", "intfloat/e5-small-v2"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = analyze(args.validation_set, backend=args.backend, model=args.model)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"output: {args.output}")


if __name__ == "__main__":
    main()
