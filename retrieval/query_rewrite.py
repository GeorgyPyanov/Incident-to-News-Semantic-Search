from __future__ import annotations

import re


TERM_EXPANSIONS = {
    "down": ("outage", "unavailable", "incident"),
    "failure": ("error", "incident", "degradation"),
    "failures": ("error", "incident", "degradation"),
    "timeout": ("latency", "unavailable", "incident"),
    "timeouts": ("latency", "unavailable", "incident"),
    "delivery": ("statuspage", "incident"),
    "affected": ("advisory", "vulnerability", "security"),
    "release": ("version", "tag", "github"),
}

PROVIDER_HINTS = (
    "github",
    "twilio",
    "cloudflare",
    "openai",
    "discord",
    "reddit",
    "datadog",
    "atlassian",
    "vercel",
    "netlify",
    "supabase",
    "anthropic",
    "shopify",
    "zoom",
)

TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_-]{1,}", re.IGNORECASE)


def rewrite_incident_query(text: str) -> str:
    """Expand terse incident logs into a document-search-friendly query."""

    tokens = [match.group(0).lower() for match in TOKEN_RE.finditer(text or "")]
    expanded: list[str] = []
    seen: set[str] = set()

    for token in tokens:
        if token not in seen:
            expanded.append(token)
            seen.add(token)
        for extra in TERM_EXPANSIONS.get(token, ()):
            if extra not in seen:
                expanded.append(extra)
                seen.add(extra)

    lowered = " ".join(tokens)
    if any(provider in lowered for provider in PROVIDER_HINTS):
        for extra in ("status", "statuspage", "incident"):
            if extra not in seen:
                expanded.append(extra)
                seen.add(extra)

    return " ".join(expanded) or text
