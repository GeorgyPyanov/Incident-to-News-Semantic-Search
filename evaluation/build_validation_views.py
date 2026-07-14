from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


DEFAULT_INPUT = Path("evaluation/data/validation_set.json")
DEFAULT_LINKED = Path("evaluation/data/validation_linked.json")
DEFAULT_BLIND = Path("evaluation/data/validation_blind.json")
DEFAULT_QRELS = Path("evaluation/data/qrels.jsonl")

ID_PATTERN = re.compile(
    r"\b(?:GHSA|CVE|RUSTSEC|PYSEC|GO)-[A-Za-z0-9_.-]+\b|"
    r"\b[0-9a-f]{8}-[0-9a-f-]{20,}\b",
    re.IGNORECASE,
)
STATUS_PREFIX_PATTERN = re.compile(r"^(.+?)\s+\[[^\]]+\]:\s*(.+)$", re.DOTALL)
OSV_PATTERN = re.compile(r"^(?P<ecosystem>[^/]+)/(?P<package>.+?)\s+affected by\s+[^:]+:\s*(?P<title>.+)$")
GITHUB_RELEASE_PATTERN = re.compile(r"^(?P<repo>[\w.-]+/[\w.-]+)\s+release published:\s*(?P<rest>.+)$", re.DOTALL)
TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_-]{2,}", re.IGNORECASE)


def build_views(input_path: Path, linked_path: Path, blind_path: Path, qrels_path: Path) -> None:
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    linked = {
        **payload,
        "validation_type": "linked",
        "description": "Linked validation set with strong source-specific relevance labels.",
    }
    blind_examples = [_blind_example(example) for example in payload["examples"]]
    blind = {
        **payload,
        "validation_type": "blind",
        "description": "Blind validation set with direct IDs and title-like prefixes removed from queries.",
        "examples": blind_examples,
    }

    linked_path.write_text(json.dumps(linked, ensure_ascii=False, indent=2), encoding="utf-8")
    blind_path.write_text(json.dumps(blind, ensure_ascii=False, indent=2), encoding="utf-8")
    qrels_path.write_text("\n".join(_qrels_lines(blind_examples)) + "\n", encoding="utf-8")


def _blind_example(example: dict[str, Any]) -> dict[str, Any]:
    copied = json.loads(json.dumps(example, ensure_ascii=False))
    message = copied["query"]["message"]
    copied["query"]["original_message"] = message
    copied["query"]["message"] = _blind_message(copied["query"]["dataset"], message)
    copied["query"]["validation_view"] = "blind"
    return copied


def _blind_message(dataset: str, message: str) -> str:
    text = ID_PATTERN.sub("security advisory", message or "")
    if dataset == "statuspage_incidents":
        match = STATUS_PREFIX_PATTERN.match(text)
        if match:
            text = match.group(2)
    elif dataset == "osv_advisories":
        match = OSV_PATTERN.match(text)
        if match:
            ecosystem = match.group("ecosystem")
            package = match.group("package")
            title = _keep_content_words(match.group("title"), max_words=8)
            text = f"{ecosystem} package {package} security vulnerability advisory {title}"
    elif dataset == "gharchive_open_source":
        match = GITHUB_RELEASE_PATTERN.match(text)
        if match:
            repo = match.group("repo")
            release_notes = _keep_content_words(match.group("rest"), max_words=16)
            text = f"GitHub repository {repo} published a software release {release_notes}"
    return " ".join(text.split())


def _keep_content_words(text: str, max_words: int) -> str:
    words = TOKEN_RE.findall(text or "")
    return " ".join(words[:max_words])


def _qrels_lines(examples: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    for example in examples:
        query_id = example["id"]
        positive = example["relevant_news"][0]
        lines.append(
            json.dumps(
                {
                    "query_id": query_id,
                    "doc_id": positive["news_id"],
                    "relevance": 3,
                    "reason": positive["relevance_reason"],
                },
                ensure_ascii=False,
            )
        )
        positive_tokens = set(TOKEN_RE.findall((positive.get("title") or "").lower()))
        for negative in example.get("negative_news", []):
            negative_tokens = set(TOKEN_RE.findall((negative.get("title") or "").lower()))
            overlap = len(positive_tokens & negative_tokens) / max(1, len(positive_tokens))
            related = negative.get("source") == positive.get("source") or overlap >= 0.25
            lines.append(
                json.dumps(
                    {
                        "query_id": query_id,
                        "doc_id": negative["news_id"],
                        "relevance": 1 if related else 0,
                        "reason": "topically_related_hard_negative" if related else negative["negative_reason"],
                    },
                    ensure_ascii=False,
                )
            )
    return lines


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build linked/blind validation views and graded qrels.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--linked-output", type=Path, default=DEFAULT_LINKED)
    parser.add_argument("--blind-output", type=Path, default=DEFAULT_BLIND)
    parser.add_argument("--qrels-output", type=Path, default=DEFAULT_QRELS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_views(args.input, args.linked_output, args.blind_output, args.qrels_output)
    print(f"linked: {args.linked_output}")
    print(f"blind: {args.blind_output}")
    print(f"qrels: {args.qrels_output}")


if __name__ == "__main__":
    main()
