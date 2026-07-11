from __future__ import annotations

import csv
import json
from pathlib import Path


def save_results_json(results: list[dict[str, object]], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"results": results}
    with output_path.open("w", encoding="utf-8") as results_file:
        json.dump(payload, results_file, indent=2, sort_keys=True)


def save_results_csv(results: list[dict[str, object]], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metric_keys = sorted({key for result in results for key in _metrics(result).keys()})
    fieldnames = [
        "approach_name",
        "selected_k_values",
        "num_queries",
        "average_execution_time_ms",
        "evaluation_timestamp",
        "configuration",
        *metric_keys,
    ]

    with output_path.open("w", encoding="utf-8", newline="") as results_file:
        writer = csv.DictWriter(results_file, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            row = {
                "approach_name": result["approach_name"],
                "selected_k_values": ";".join(str(k) for k in result["selected_k_values"]),
                "num_queries": result["num_queries"],
                "average_execution_time_ms": result["average_execution_time_ms"],
                "evaluation_timestamp": result["evaluation_timestamp"],
                "configuration": json.dumps(result["configuration"], sort_keys=True),
            }
            row.update(_metrics(result))
            writer.writerow(row)


def format_comparison_table(results: list[dict[str, object]]) -> str:
    if not results:
        return "No evaluation results."

    metric_keys = _table_metric_keys(results)
    headers = ["Approach", *metric_keys, "Avg ms", "Queries"]
    rows = []
    for result in results:
        metrics = _metrics(result)
        rows.append(
            [
                str(result["approach_name"]),
                *[_format_float(metrics.get(metric, 0.0)) for metric in metric_keys],
                _format_float(float(result["average_execution_time_ms"])),
                str(result["num_queries"]),
            ]
        )

    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]
    lines = [_format_row(headers, widths), _format_row(["-" * width for width in widths], widths)]
    lines.extend(_format_row(row, widths) for row in rows)
    return "\n".join(lines)


def _metrics(result: dict[str, object]) -> dict[str, float]:
    raw_metrics = result.get("metrics", {})
    if not isinstance(raw_metrics, dict):
        return {}
    return {str(key): float(value) for key, value in raw_metrics.items()}


def _table_metric_keys(results: list[dict[str, object]]) -> list[str]:
    metric_keys = sorted({key for result in results for key in _metrics(result).keys()})
    ordered_keys: list[str] = []
    for prefix in ("precision@", "recall@", "ndcg@"):
        ordered_keys.extend(key for key in metric_keys if key.startswith(prefix))
    ordered_keys.extend(key for key in ("mrr", "map") if key in metric_keys)
    return ordered_keys


def _format_row(values: list[str], widths: list[int]) -> str:
    return " | ".join(value.ljust(width) for value, width in zip(values, widths))


def _format_float(value: float) -> str:
    return f"{value:.4f}"

