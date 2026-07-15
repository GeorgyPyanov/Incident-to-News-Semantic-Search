from __future__ import annotations

import argparse
import json
import os
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

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
    quantization: str = "none",
) -> dict[str, Any]:
    payload = json.loads(validation_path.read_text(encoding="utf-8"))
    examples = payload["examples"]
    client = build_embedding_client(
        backend=backend,
        model=model,
        dimensions=settings.embedding_dim,
        quantization=quantization,
    )

    stats: list[ExampleStats] = []
    vectors: list[list[float]] = []
    labels: list[dict[str, Any]] = []
    for example in examples:
        query_text = build_query_text(str(example["query"]["message"]))
        positive_text = build_document_text(_news_text(example["relevant_news"][0]))
        negative_texts = [build_document_text(_news_text(item)) for item in example.get("negative_news", [])]
        query_vector = client.embed_text(query_text)
        positive_vector = client.embed_text(positive_text)
        negative_vectors = [client.embed_text(text) for text in negative_texts] or [positive_vector]

        vectors.append(query_vector)
        labels.append({"example_id": example["id"], "kind": "query"})
        vectors.append(positive_vector)
        labels.append({"example_id": example["id"], "kind": "positive"})
        hardest_negative_vector = max(negative_vectors, key=lambda vector: _cosine(query_vector, vector))
        vectors.append(hardest_negative_vector)
        labels.append({"example_id": example["id"], "kind": "hardest_negative"})

        stats.append(
            ExampleStats(
                query_norm=_norm(query_vector),
                positive_similarity=_cosine(query_vector, positive_vector),
                hardest_negative_similarity=_cosine(query_vector, hardest_negative_vector),
            )
        )

    margins = [item.margin for item in stats]
    positive_sims = [item.positive_similarity for item in stats]
    negative_sims = [item.hardest_negative_similarity for item in stats]
    norms = [item.query_norm for item in stats]
    separated = sum(1 for item in stats if item.correctly_separated)
    matrix = np.asarray(vectors, dtype=np.float32)
    dimension_distribution = _dimension_distribution(matrix)
    projections = _project_vectors(matrix, labels)

    return {
        "validation_set": str(validation_path),
        "backend": backend,
        "model": model,
        "quantization": quantization,
        "examples": len(stats),
        "query_embedding_norm": _summary(norms),
        "positive_similarity": _summary(positive_sims),
        "hardest_negative_similarity": _summary(negative_sims),
        "margin": _summary(margins),
        "separation_rate": separated / len(stats) if stats else 0.0,
        "dimension_distribution": dimension_distribution,
        "projections": projections,
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


def _dimension_distribution(matrix: np.ndarray) -> dict[str, Any]:
    if matrix.size == 0:
        return {"dimensions": 0}
    variances = matrix.var(axis=0)
    means = matrix.mean(axis=0)
    stds = matrix.std(axis=0)
    top_indices = np.argsort(variances)[::-1][:10]
    return {
        "dimensions": int(matrix.shape[1]),
        "value_summary": {
            "mean_abs": float(np.abs(matrix).mean()),
            "std_mean": float(stds.mean()),
            "std_p95": float(np.percentile(stds, 95)),
            "variance_mean": float(variances.mean()),
            "variance_p95": float(np.percentile(variances, 95)),
        },
        "mean_range": {
            "min": float(means.min()),
            "max": float(means.max()),
        },
        "top_variance_dimensions": [
            {"dimension": int(index), "variance": float(variances[index])}
            for index in top_indices
        ],
        "effective_rank": _effective_rank(matrix),
    }


def _project_vectors(matrix: np.ndarray, labels: list[dict[str, Any]]) -> dict[str, Any]:
    if matrix.size == 0:
        return {"pca": None, "tsne": None}
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    components = min(8, centered.shape[0], centered.shape[1])
    pca = PCA(n_components=components)
    pca_result = pca.fit_transform(centered)
    pca_summary = {
        "components": int(components),
        "explained_variance_ratio": [float(value) for value in pca.explained_variance_ratio_],
        "coordinates": [
            {
                **labels[index],
                "x": float(point[0]),
                "y": float(point[1]) if point.shape[0] > 1 else 0.0,
            }
            for index, point in enumerate(pca_result[:, :2])
        ],
    }

    if centered.shape[0] < 3:
        return {"pca": pca_summary, "tsne": None}

    tsne_input_dims = min(50, centered.shape[0] - 1, centered.shape[1])
    tsne_input = centered
    if tsne_input_dims < centered.shape[1]:
        tsne_input = PCA(n_components=tsne_input_dims).fit_transform(centered)
    perplexity = min(30, max(5, (centered.shape[0] - 1) // 3))
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=42,
    )
    tsne_result = tsne.fit_transform(tsne_input)
    tsne_summary = {
        "perplexity": int(perplexity),
        "coordinates": [
            {
                **labels[index],
                "x": float(point[0]),
                "y": float(point[1]),
            }
            for index, point in enumerate(tsne_result)
        ],
    }
    return {"pca": pca_summary, "tsne": tsne_summary}


def _effective_rank(matrix: np.ndarray) -> float:
    singular_values = np.linalg.svd(matrix - matrix.mean(axis=0, keepdims=True), compute_uv=False)
    total = singular_values.sum()
    if not total:
        return 0.0
    probabilities = singular_values / total
    entropy = -float(np.sum(probabilities * np.log(probabilities + 1e-12)))
    return float(np.exp(entropy))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze the embedding space on the validation set.")
    parser.add_argument("--validation-set", type=Path, default=DEFAULT_VALIDATION_SET)
    parser.add_argument("--backend", default="auto")
    parser.add_argument("--model", default=os.getenv("EMBEDDING_MODEL", "intfloat/e5-small-v2"))
    parser.add_argument("--quantization", default=os.getenv("EMBEDDING_QUANTIZATION", "none"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = analyze(args.validation_set, backend=args.backend, model=args.model, quantization=args.quantization)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"output: {args.output}")


if __name__ == "__main__":
    main()
