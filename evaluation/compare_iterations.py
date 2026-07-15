from __future__ import annotations

import argparse
import json
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from data.embed_structured_events import embed_structured_events
from database.settings import settings
from evaluation.benchmark_real import (
    DEFAULT_SEARCHES,
    DEFAULT_WARMUPS,
    benchmark_document_sample,
    collect_platform_info,
    default_connection_factory,
    default_document_embedder,
    default_document_sample_loader,
    run_real_benchmark,
)
from evaluation.embedding_analysis import analyze as analyze_embeddings
from evaluation.validate_qrels import (
    DEFAULT_QRELS,
    DEFAULT_VALIDATION_SET as DEFAULT_BLIND_VALIDATION_SET,
    evaluate as evaluate_qrels,
)
from evaluation.validate_retrieval import (
    DEFAULT_VALIDATION_SET as DEFAULT_LINKED_VALIDATION_SET,
    evaluate as evaluate_linked,
)


DEFAULT_OUTPUT = Path("evaluation/iteration_comparison_results.json")


@dataclass(frozen=True)
class IterationConfig:
    name: str
    embedding_backend: str
    embedding_model: str
    embedding_quantization: str
    fusion_mode: str
    refresh_embeddings: bool

    def env(self) -> dict[str, str]:
        return {
            "EMBEDDING_BACKEND": self.embedding_backend,
            "EMBEDDING_MODEL": self.embedding_model,
            "EMBEDDING_QUANTIZATION": self.embedding_quantization,
            "RETRIEVAL_FUSION_MODE": self.fusion_mode,
        }


def compare_iterations(
    *,
    baseline: IterationConfig,
    candidate: IterationConfig,
    linked_validation_path: Path = DEFAULT_LINKED_VALIDATION_SET,
    blind_validation_path: Path = DEFAULT_BLIND_VALIDATION_SET,
    qrels_path: Path = DEFAULT_QRELS,
    output_path: Path = DEFAULT_OUTPUT,
    top_k: int = 10,
    searches: int = DEFAULT_SEARCHES,
    warmups: int = DEFAULT_WARMUPS,
    embedding_sample_size: int = 100,
    save: bool = True,
) -> dict[str, Any]:
    hardware = collect_platform_info()
    iterations = [
        run_iteration(
            config=baseline,
            linked_validation_path=linked_validation_path,
            blind_validation_path=blind_validation_path,
            qrels_path=qrels_path,
            top_k=top_k,
            searches=searches,
            warmups=warmups,
            embedding_sample_size=embedding_sample_size,
            hardware=hardware,
        ),
        run_iteration(
            config=candidate,
            linked_validation_path=linked_validation_path,
            blind_validation_path=blind_validation_path,
            qrels_path=qrels_path,
            top_k=top_k,
            searches=searches,
            warmups=warmups,
            embedding_sample_size=embedding_sample_size,
            hardware=hardware,
        ),
    ]
    result = {
        "benchmark_type": "retrieval_iteration_comparison",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "linked_validation_set": str(linked_validation_path),
        "blind_validation_set": str(blind_validation_path),
        "qrels": str(qrels_path),
        "top_k": top_k,
        "platform": hardware,
        "iterations": iterations,
        "delta": _delta(iterations[0], iterations[1]),
        "database_state_after_run": candidate.name if candidate.refresh_embeddings else baseline.name,
    }
    if save:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def run_iteration(
    *,
    config: IterationConfig,
    linked_validation_path: Path,
    blind_validation_path: Path,
    qrels_path: Path,
    top_k: int,
    searches: int,
    warmups: int,
    embedding_sample_size: int,
    hardware: dict[str, Any],
) -> dict[str, Any]:
    with temporary_environment(config.env()):
        refresh_result = {
            "requested": config.refresh_embeddings,
            "documents_refreshed": 0,
            "total_seconds": None,
            "throughput_documents_per_second": None,
        }
        if config.refresh_embeddings:
            started = time.perf_counter()
            refreshed = embed_structured_events(limit=None, refresh=True)
            elapsed = time.perf_counter() - started
            refresh_result = {
                "requested": True,
                "documents_refreshed": refreshed,
                "total_seconds": elapsed,
                "throughput_documents_per_second": refreshed / elapsed if elapsed > 0 else None,
            }

        search_benchmark = run_real_benchmark(
            validation_path=blind_validation_path,
            output_path=Path(os.devnull),
            top_k=top_k,
            searches=searches,
            warmups=warmups,
            rebuild_index=True,
            hardware=hardware,
            save=False,
        )
        document_embedding_benchmark = run_document_embedding_benchmark(
            sample_size=embedding_sample_size,
            hardware=hardware,
        )
        linked_quality = evaluate_linked(linked_validation_path, top_k)
        qrels_quality = evaluate_qrels(blind_validation_path, qrels_path, top_k)
        vector_analysis = analyze_embeddings(
            blind_validation_path,
            backend=config.embedding_backend,
            model=config.embedding_model,
            quantization=config.embedding_quantization,
        )

    return {
        "name": config.name,
        "configuration": {
            "embedding_backend": config.embedding_backend,
            "embedding_model": config.embedding_model,
            "embedding_quantization": config.embedding_quantization,
            "embedding_dimension": settings.embedding_dim,
            "fusion_mode": config.fusion_mode,
        },
        "embedding_refresh": refresh_result,
        "index": search_benchmark["index"],
        "search": search_benchmark["search"],
        "explain_analyze": search_benchmark["explain_analyze"],
        "document_embedding_benchmark": document_embedding_benchmark,
        "linked_validation": linked_quality,
        "qrels_validation": qrels_quality,
        "embedding_analysis": vector_analysis,
    }


def run_document_embedding_benchmark(
    *,
    sample_size: int,
    hardware: dict[str, Any],
) -> dict[str, Any]:
    with default_connection_factory() as connection:
        with connection.cursor() as cursor:
            cursor.execute("SET TRANSACTION READ ONLY")
            sample = default_document_sample_loader(cursor, sample_size)
            return benchmark_document_sample(
                sample,
                dimension=settings.embedding_dim,
                embedder=default_document_embedder,
                hardware=hardware,
            )


@contextmanager
def temporary_environment(values: dict[str, str]) -> Iterator[None]:
    previous = {key: os.environ.get(key) for key in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _delta(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "qrels_hybrid_ndcg_at_k": _metric_delta(
            baseline,
            candidate,
            ("qrels_validation", "modes", "hybrid", "ndcg_at_k"),
        ),
        "qrels_hybrid_mrr_at_k": _metric_delta(
            baseline,
            candidate,
            ("qrels_validation", "modes", "hybrid", "mrr_at_k"),
        ),
        "linked_hybrid_ndcg_at_k": _metric_delta(
            baseline,
            candidate,
            ("linked_validation", "modes", "hybrid", "ndcg_at_k"),
        ),
        "document_embedding_mean_ms": _metric_delta(
            baseline,
            candidate,
            ("document_embedding_benchmark", "per_document_latency_ms", "mean"),
        ),
        "pgvector_search_mean_ms": _metric_delta(
            baseline,
            candidate,
            ("search", "database_index_search_latency_ms", "mean"),
        ),
        "end_to_end_retrieval_mean_ms": _metric_delta(
            baseline,
            candidate,
            ("search", "end_to_end_retrieval_latency_ms", "mean"),
        ),
    }


def _metric_delta(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    path: tuple[str, ...],
) -> dict[str, float | None]:
    baseline_value = _nested_float(baseline, path)
    candidate_value = _nested_float(candidate, path)
    if baseline_value is None or candidate_value is None:
        return {"baseline": baseline_value, "candidate": candidate_value, "absolute": None, "relative": None}
    absolute = candidate_value - baseline_value
    relative = absolute / baseline_value if baseline_value else None
    return {
        "baseline": baseline_value,
        "candidate": candidate_value,
        "absolute": absolute,
        "relative": relative,
    }


def _nested_float(payload: dict[str, Any], path: tuple[str, ...]) -> float | None:
    current: Any = payload
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    try:
        return float(current)
    except (TypeError, ValueError):
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two retrieval iterations on the same validation sets.")
    parser.add_argument("--linked-validation-set", type=Path, default=DEFAULT_LINKED_VALIDATION_SET)
    parser.add_argument("--blind-validation-set", type=Path, default=DEFAULT_BLIND_VALIDATION_SET)
    parser.add_argument("--qrels", type=Path, default=DEFAULT_QRELS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--searches", type=int, default=DEFAULT_SEARCHES)
    parser.add_argument("--warmups", type=int, default=DEFAULT_WARMUPS)
    parser.add_argument("--embedding-sample-size", type=int, default=100)
    parser.add_argument("--backend", default=os.getenv("EMBEDDING_BACKEND", "sentence-transformer"))
    parser.add_argument("--baseline-model", default=os.getenv("EMBEDDING_MODEL", "intfloat/e5-small-v2"))
    parser.add_argument("--baseline-quantization", default="none")
    parser.add_argument("--baseline-fusion-mode", default="rrf")
    parser.add_argument("--refresh-baseline", action="store_true")
    parser.add_argument("--candidate-model", default=os.getenv("EMBEDDING_MODEL", "intfloat/e5-small-v2"))
    parser.add_argument("--candidate-quantization", default="dynamic")
    parser.add_argument("--candidate-fusion-mode", default="normalized_sum")
    parser.add_argument(
        "--skip-candidate-refresh",
        action="store_true",
        help="Do not overwrite stored embeddings before measuring the candidate iteration.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline = IterationConfig(
        name="iteration_1_baseline_e5_hnsw_bm25_rrf",
        embedding_backend=args.backend,
        embedding_model=args.baseline_model,
        embedding_quantization=args.baseline_quantization,
        fusion_mode=args.baseline_fusion_mode,
        refresh_embeddings=args.refresh_baseline,
    )
    candidate = IterationConfig(
        name="iteration_2_quantized_embedding_normalized_sum",
        embedding_backend=args.backend,
        embedding_model=args.candidate_model,
        embedding_quantization=args.candidate_quantization,
        fusion_mode=args.candidate_fusion_mode,
        refresh_embeddings=not args.skip_candidate_refresh,
    )
    result = compare_iterations(
        baseline=baseline,
        candidate=candidate,
        linked_validation_path=args.linked_validation_set,
        blind_validation_path=args.blind_validation_set,
        qrels_path=args.qrels,
        output_path=args.output,
        top_k=args.top_k,
        searches=args.searches,
        warmups=args.warmups,
        embedding_sample_size=args.embedding_sample_size,
    )
    print(json.dumps(result["delta"], ensure_ascii=False, indent=2))
    print(f"output: {args.output}")


if __name__ == "__main__":
    main()
