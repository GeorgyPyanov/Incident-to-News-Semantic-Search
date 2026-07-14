from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
import platform
import statistics
import sys
import tempfile
import time
from collections.abc import Callable, Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evaluation.dataset import DEFAULT_DATASET_PATH, EvaluationDataset, load_evaluation_dataset
from evaluation.retrievers import article_embedding_text, cosine_similarity, generate_text_embedding
from retrieval.schemas import NewsArticle


DEFAULT_OUTPUT = Path("evaluation/benchmark_results.json")
DEFAULT_DIMENSIONS = 32
DEFAULT_SEARCHES = 100
MODEL_NAME = "deterministic_hashing_token_average"
INDEX_TYPE = "exact_cosine_flat"


class ExactCosineIndex:
    """Small exact index built from the project's deterministic embeddings."""

    def __init__(self, ids: Iterable[str], vectors: Iterable[tuple[float, ...]]) -> None:
        self.ids = tuple(ids)
        self.vectors = tuple(tuple(vector) for vector in vectors)
        if len(self.ids) != len(self.vectors):
            raise ValueError("Index IDs and vectors must have the same length")

    def search(self, query_vector: tuple[float, ...], top_k: int) -> list[tuple[str, float]]:
        scores = [
            (article_id, cosine_similarity(query_vector, vector))
            for article_id, vector in zip(self.ids, self.vectors)
        ]
        scores.sort(key=lambda item: (-item[1], item[0]))
        return scores[:top_k]

    def save(self, path: Path) -> None:
        path.write_text(
            json.dumps({"ids": self.ids, "vectors": self.vectors}, separators=(",", ":")),
            encoding="utf-8",
        )


def run_benchmark(
    dataset: EvaluationDataset,
    *,
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
    output_path: str | Path = DEFAULT_OUTPUT,
    dimensions: int = DEFAULT_DIMENSIONS,
    top_k: int = 10,
    searches: int = DEFAULT_SEARCHES,
    timestamp: str | None = None,
    hardware: dict[str, Any] | None = None,
    clock: Callable[[], float] = time.perf_counter,
    save: bool = True,
) -> dict[str, Any]:
    if searches < 100:
        raise ValueError("searches must be at least 100")
    if dimensions <= 0:
        raise ValueError("dimensions must be positive")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if not dataset.queries:
        raise ValueError("dataset must contain at least one query")

    articles = _unique_articles(dataset)

    embedding_started = clock()
    vectors = [generate_text_embedding(article_embedding_text(article), dimensions) for article in articles]
    embedding_seconds = clock() - embedding_started

    index_started = clock()
    index = ExactCosineIndex((article.id for article in articles), vectors)
    index_seconds = clock() - index_started

    output = Path(output_path)
    disk_size_bytes = _serialized_index_size(index, output.parent)
    memory_size_bytes = _deep_size(index)

    # Everything needed for search is initialized before the warm-up and timed loop.
    queries = tuple(query.incident_log for query in dataset.queries)
    _search(index, queries[0], dimensions, top_k)

    latencies_ms: list[float] = []
    for search_number in range(searches):
        query = queries[search_number % len(queries)]
        started = clock()
        _search(index, query, dimensions, top_k)
        latencies_ms.append((clock() - started) * 1000.0)

    result: dict[str, Any] = {
        "timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
        "platform": hardware or collect_platform_info(),
        "dataset": {
            "path": str(dataset_path),
            "queries": len(dataset.queries),
            "articles": len(articles),
        },
        "model_name": MODEL_NAME,
        "index_type": INDEX_TYPE,
        "top_k": top_k,
        "embeddings": {
            "generation_time_seconds": embedding_seconds,
            "count": len(vectors),
            "dimension": dimensions,
        },
        "index": {
            "build_time_seconds": index_seconds,
            "size_on_disk_bytes": disk_size_bytes,
            "size_in_memory_bytes": memory_size_bytes,
        },
        "search": {
            "warmup_searches": 1,
            "benchmark_searches": searches,
            "clock": "time.perf_counter",
            "latency_ms": _latency_statistics(latencies_ms),
        },
    }

    if save:
        save_results(result, output)
    return result


def save_results(results: dict[str, Any], output_path: str | Path = DEFAULT_OUTPUT) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def collect_platform_info() -> dict[str, Any]:
    logical_cores = os.cpu_count()
    return {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "python_version": platform.python_version(),
        "cpu": {
            "model": _cpu_model(),
            "logical_cores": logical_cores,
            "physical_cores": _physical_core_count(),
        },
        "total_ram_bytes": _total_ram_bytes(),
        "gpu": _gpu_info(),
    }


def format_summary(results: dict[str, Any], output_path: str | Path) -> str:
    hardware = results["platform"]
    cpu = hardware["cpu"]
    gpu = hardware["gpu"]
    embeddings = results["embeddings"]
    index = results["index"]
    search = results["search"]
    latency = search["latency_ms"]
    gpu_line = "No GPU" if gpu["model"] == "No GPU" else f"{gpu['model']} ({_format_bytes(gpu['vram_bytes'])} VRAM)"
    return "\n".join(
        (
            "Retrieval benchmark",
            f"  Timestamp: {results['timestamp']}",
            f"  Platform: {hardware['system']} {hardware['release']} ({hardware['machine']})",
            f"  CPU: {cpu['model']} ({cpu['logical_cores']} logical cores)",
            f"  RAM: {_format_bytes(hardware['total_ram_bytes'])}",
            f"  GPU: {gpu_line}",
            f"  Dataset: {results['dataset']['queries']} queries, {results['dataset']['articles']} articles",
            f"  Model: {results['model_name']} ({embeddings['dimension']} dimensions)",
            f"  Embeddings: {embeddings['count']} in {embeddings['generation_time_seconds']:.6f} s",
            f"  Index: {results['index_type']} built in {index['build_time_seconds']:.6f} s",
            f"  Index size: {_format_bytes(index['size_on_disk_bytes'])} on disk, {_format_bytes(index['size_in_memory_bytes'])} in memory",
            f"  Search: top-{results['top_k']}, 1 warm-up + {search['benchmark_searches']} measured",
            "  Latency (ms): "
            f"mean={latency['mean']:.3f}, median={latency['median']:.3f}, "
            f"p95={latency['p95']:.3f}, min={latency['min']:.3f}, max={latency['max']:.3f}",
            f"  Results: {output_path}",
        )
    )


def _search(index: ExactCosineIndex, query: str, dimensions: int, top_k: int) -> list[tuple[str, float]]:
    return index.search(generate_text_embedding(query, dimensions), top_k)


def _unique_articles(dataset: EvaluationDataset) -> tuple[NewsArticle, ...]:
    by_id: dict[str, NewsArticle] = {}
    for query in dataset.queries:
        for article in query.candidate_articles:
            by_id.setdefault(article.id, article)
    return tuple(by_id.values())


def _latency_statistics(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "mean": statistics.fmean(ordered),
        "median": statistics.median(ordered),
        "p95": _percentile(ordered, 0.95),
        "min": ordered[0],
        "max": ordered[-1],
    }


def _percentile(ordered: list[float], fraction: float) -> float:
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _serialized_index_size(index: ExactCosineIndex, directory: Path) -> int:
    directory.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(prefix="benchmark-index-", suffix=".json", dir=directory, delete=False) as file:
            temporary_path = Path(file.name)
        index.save(temporary_path)
        return temporary_path.stat().st_size
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _deep_size(value: object) -> int:
    seen: set[int] = set()

    def measure(item: object) -> int:
        item_id = id(item)
        if item_id in seen:
            return 0
        seen.add(item_id)
        size = sys.getsizeof(item)
        if isinstance(item, dict):
            size += sum(measure(key) + measure(child) for key, child in item.items())
        elif isinstance(item, (tuple, list, set, frozenset)):
            size += sum(measure(child) for child in item)
        elif hasattr(item, "__dict__"):
            size += measure(vars(item))
        return size

    return measure(value)


def _cpu_model() -> str:
    if platform.system() == "Windows":
        try:
            import winreg

            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            ) as key:
                registry_model = str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
                if registry_model:
                    return registry_model
        except (ImportError, OSError):
            pass
    model = platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER", "")
    if not model and Path("/proc/cpuinfo").exists():
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.lower().startswith("model name"):
                model = line.partition(":")[2].strip()
                break
    return model or "Unknown CPU"


def _physical_core_count() -> int | None:
    try:
        import psutil  # type: ignore[import-not-found]

        return psutil.cpu_count(logical=False)
    except (ImportError, OSError):
        pass
    if platform.system() == "Windows":
        try:
            class ProcessorInfoUnion(ctypes.Union):
                _fields_ = [("reserved", ctypes.c_ulonglong * 2)]

            class LogicalProcessorInfo(ctypes.Structure):
                _fields_ = [
                    ("processor_mask", ctypes.c_size_t),
                    ("relationship", ctypes.c_int),
                    ("data", ProcessorInfoUnion),
                ]

            length = ctypes.c_ulong(0)
            function = ctypes.windll.kernel32.GetLogicalProcessorInformation
            function(None, ctypes.byref(length))
            count = length.value // ctypes.sizeof(LogicalProcessorInfo)
            entries = (LogicalProcessorInfo * count)()
            if function(entries, ctypes.byref(length)):
                physical = sum(1 for entry in entries if entry.relationship == 0)
                return physical or None
        except (AttributeError, OSError, ValueError):
            pass
    return None


def _total_ram_bytes() -> int | None:
    try:
        import psutil  # type: ignore[import-not-found]

        return int(psutil.virtual_memory().total)
    except (ImportError, OSError):
        pass

    if platform.system() == "Windows":
        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("length", ctypes.c_ulong),
                ("memory_load", ctypes.c_ulong),
                ("total_physical", ctypes.c_ulonglong),
                ("available_physical", ctypes.c_ulonglong),
                ("total_page_file", ctypes.c_ulonglong),
                ("available_page_file", ctypes.c_ulonglong),
                ("total_virtual", ctypes.c_ulonglong),
                ("available_virtual", ctypes.c_ulonglong),
                ("available_extended_virtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.length = ctypes.sizeof(MemoryStatus)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.total_physical)
    try:
        return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    except (AttributeError, OSError, ValueError):
        return None


def _gpu_info() -> dict[str, Any]:
    try:
        import torch

        if torch.cuda.is_available():
            properties = torch.cuda.get_device_properties(0)
            return {"model": properties.name, "vram_bytes": int(properties.total_memory)}
    except (ImportError, RuntimeError, OSError):
        pass
    return {"model": "No GPU", "vram_bytes": None}


def _format_bytes(value: int | None) -> str:
    if value is None:
        return "Unknown"
    size = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024.0 or unit == "TiB":
            return f"{size:.2f} {unit}"
        size /= 1024.0
    raise AssertionError("unreachable")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark the in-memory semantic retrieval pipeline.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dimensions", type=int, default=DEFAULT_DIMENSIONS)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--searches", type=int, default=DEFAULT_SEARCHES)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = load_evaluation_dataset(args.dataset)
    results = run_benchmark(
        dataset,
        dataset_path=args.dataset,
        output_path=args.output,
        dimensions=args.dimensions,
        top_k=args.top_k,
        searches=args.searches,
    )
    print(format_summary(results, args.output))


if __name__ == "__main__":
    main()
