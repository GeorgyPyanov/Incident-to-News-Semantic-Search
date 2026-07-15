import argparse
import csv
import gzip
import hashlib
import json
import re
import zipfile
from datetime import UTC, datetime, timedelta
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen, urlretrieve

import psycopg
import feedparser
from datasets import load_dataset

from database.migrate import _postgres_url, apply_migrations
from database.settings import settings


DEFAULT_NEWS_DATASET = "fancyzhx/ag_news"
DEFAULT_HF_LOGHUB_DATASET = "bolu61/loghub_2"
DEFAULT_LOG_URL = "https://raw.githubusercontent.com/logpai/loghub/master/Apache/Apache_2k.log"
DEFAULT_GDELT_URL = "http://data.gdeltproject.org/events/2005.zip"
DEFAULT_GDELT_CACHE = Path("data/cache/gdelt_2005.zip")
DEFAULT_GDELT_V2_MASTER_URL = "http://data.gdeltproject.org/gdeltv2/masterfilelist.txt"
DEFAULT_GHARCHIVE_BASE_URL = "https://data.gharchive.org"
DEFAULT_HN_SEARCH_URL = "https://hn.algolia.com/api/v1/search_by_date"
DEFAULT_GOOGLE_NEWS_RSS_URL = "https://news.google.com/rss/search"
DEFAULT_OSV_QUERY_URL = "https://api.osv.dev/v1/query"
GDELT_2005_SOURCE_NAME = "gdelt_2005_events"
GDELTV2_SOURCE_NAME = "gdeltv2_recent_events"
PROFILE_DEFAULT_MINIMUM_TOTAL = 50_000
LARGE_IMPORT_PRESET = {
    "minimum_total": 500_000,
    "news_limit": 0,
    "logs_limit": 0,
    "hf_loghub_limit": 0,
    "statuspage_limit": 5_000,
    "hn_limit_per_provider": 100,
    "google_news_limit_per_provider": 100,
    "osv_limit_per_package": 100,
    "gdelt_limit": 0,
    "gdeltv2_limit": 400_000,
    "gdeltv2_max_files": 600,
    "gdeltv2_start": datetime(2026, 6, 1, tzinfo=UTC),
    "gdeltv2_end": datetime(2026, 7, 14, tzinfo=UTC),
    "gdeltv2_max_files_per_day": 8,
    "gharchive_projects": 400,
    "gharchive_log_limit": 100_000,
    "gharchive_news_limit": 5_000,
}
PROFILE_PRESETS = {
    "large": LARGE_IMPORT_PRESET,
    "project_500k": LARGE_IMPORT_PRESET,
}
STATUSPAGE_SOURCES = {
    "GitHub": "https://www.githubstatus.com/api/v2/incidents.json",
    "Cloudflare": "https://www.cloudflarestatus.com/api/v2/incidents.json",
    "OpenAI": "https://status.openai.com/api/v2/incidents.json",
    "Discord": "https://discordstatus.com/api/v2/incidents.json",
    "Reddit": "https://www.redditstatus.com/api/v2/incidents.json",
    "Datadog": "https://status.datadoghq.com/api/v2/incidents.json",
    "Atlassian": "https://status.atlassian.com/api/v2/incidents.json",
    "Twilio": "https://status.twilio.com/api/v2/incidents.json",
    "SendGrid": "https://status.sendgrid.com/api/v2/incidents.json",
    "DigitalOcean": "https://status.digitalocean.com/api/v2/incidents.json",
    "Vercel": "https://www.vercel-status.com/api/v2/incidents.json",
    "Netlify": "https://www.netlifystatus.com/api/v2/incidents.json",
    "Supabase": "https://status.supabase.com/api/v2/incidents.json",
    "Anthropic": "https://status.anthropic.com/api/v2/incidents.json",
    "Shopify": "https://www.shopifystatus.com/api/v2/incidents.json",
    "Zoom": "https://status.zoom.us/api/v2/incidents.json",
    "Dropbox": "https://status.dropbox.com/api/v2/incidents.json",
    "Box": "https://status.box.com/api/v2/incidents.json",
    "Sentry": "https://status.sentry.io/api/v2/incidents.json",
    "CircleCI": "https://status.circleci.com/api/v2/incidents.json",
    "HashiCorp": "https://status.hashicorp.com/api/v2/incidents.json",
    "MongoDB": "https://status.mongodb.com/api/v2/incidents.json",
    "Confluent": "https://status.confluent.cloud/api/v2/incidents.json",
    "Elastic": "https://status.elastic.co/api/v2/incidents.json",
    "npm": "https://status.npmjs.org/api/v2/incidents.json",
    "PyPI": "https://status.python.org/api/v2/incidents.json",
    "Grafana": "https://status.grafana.com/api/v2/incidents.json",
    "NewRelic": "https://status.newrelic.com/api/v2/incidents.json",
    "Snowflake": "https://status.snowflake.com/api/v2/incidents.json",
    "Figma": "https://status.figma.com/api/v2/incidents.json",
    "Canva": "https://www.canvastatus.com/api/v2/incidents.json",
    "Zapier": "https://status.zapier.com/api/v2/incidents.json",
    "HubSpot": "https://status.hubspot.com/api/v2/incidents.json",
    "DropboxSign": "https://status.hellosign.com/api/v2/incidents.json",
    "Miro": "https://status.miro.com/api/v2/incidents.json",
    "Airtable": "https://status.airtable.com/api/v2/incidents.json",
    "LaunchDarkly": "https://status.launchdarkly.com/api/v2/incidents.json",
    "1Password": "https://status.1password.com/api/v2/incidents.json",
    "Bitbucket": "https://bitbucket.status.atlassian.com/api/v2/incidents.json",
    "Jira": "https://jira-software.status.atlassian.com/api/v2/incidents.json",
    "Confluence": "https://confluence.status.atlassian.com/api/v2/incidents.json",
}

OSV_PACKAGES = [
    ("PyPI", "django"),
    ("PyPI", "flask"),
    ("PyPI", "requests"),
    ("PyPI", "urllib3"),
    ("PyPI", "pillow"),
    ("PyPI", "numpy"),
    ("PyPI", "tensorflow"),
    ("PyPI", "torch"),
    ("PyPI", "fastapi"),
    ("PyPI", "jinja2"),
    ("npm", "lodash"),
    ("npm", "express"),
    ("npm", "react"),
    ("npm", "next"),
    ("npm", "axios"),
    ("npm", "webpack"),
    ("npm", "moment"),
    ("npm", "minimist"),
    ("npm", "serialize-javascript"),
    ("npm", "jsonwebtoken"),
    ("Go", "github.com/gin-gonic/gin"),
    ("Go", "github.com/golang/protobuf"),
    ("Go", "golang.org/x/crypto"),
    ("Go", "google.golang.org/grpc"),
    ("Go", "github.com/kubernetes/kubernetes"),
    ("Maven", "org.springframework:spring-core"),
    ("Maven", "org.springframework.boot:spring-boot"),
    ("Maven", "org.apache.logging.log4j:log4j-core"),
    ("Maven", "com.fasterxml.jackson.core:jackson-databind"),
    ("Maven", "org.apache.struts:struts2-core"),
    ("crates.io", "tokio"),
    ("crates.io", "hyper"),
    ("crates.io", "openssl"),
    ("crates.io", "serde"),
    ("crates.io", "time"),
]

NEWS_LABELS = {
    0: "World",
    1: "Sports",
    2: "Business",
    3: "Sci/Tech",
}
APACHE_TIME_RE = re.compile(r"^\[(?P<value>[A-Z][a-z]{2} [A-Z][a-z]{2} \d{2} \d{2}:\d{2}:\d{2} \d{4})\]")


def sha256_hex(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def stable_int_id(value: str) -> int:
    return int(sha256_hex(value)[:15], 16) % 2_000_000_000


def parse_hour(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%dT%H").replace(tzinfo=UTC)


def parse_date(value: str) -> datetime:
    return parse_datetime(value)


def parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    normalized = value.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def apply_profile(args: argparse.Namespace) -> int:
    profile_name = str(getattr(args, "profile", "default") or "default")
    preset = PROFILE_PRESETS.get(profile_name)
    if preset is None:
        return int(getattr(args, "minimum_total", PROFILE_DEFAULT_MINIMUM_TOTAL))

    for key, value in preset.items():
        setattr(args, key, value)
    return int(preset["minimum_total"])


def news_title_and_body(text: str) -> tuple[str, str | None]:
    normalized = " ".join(text.split())
    if not normalized:
        return "Untitled news item", None
    if ". " in normalized:
        title, body = normalized.split(". ", 1)
        return title[:500], body
    return normalized[:500], None


def detect_severity(message: str) -> str | None:
    match = re.search(r"\b(emerg|alert|crit|critical|error|err|warn|warning|notice|info|debug)\b", message, re.I)
    if not match:
        return None
    value = match.group(1).lower()
    if value == "err":
        return "error"
    if value == "critical":
        return "crit"
    if value == "warning":
        return "warn"
    return value


def parse_apache_event_time(message: str) -> datetime | None:
    match = APACHE_TIME_RE.match(message)
    if not match:
        return None
    return datetime.strptime(match.group("value"), "%a %b %d %H:%M:%S %Y").replace(tzinfo=UTC)


def iter_remote_lines(url: str, limit: int):
    with urlopen(url, timeout=60) as response:
        emitted = 0
        for line_number, raw_line in enumerate(response, start=1):
            message = raw_line.decode("utf-8", errors="replace").strip()
            if not message:
                continue
            yield line_number, message
            emitted += 1
            if emitted >= limit:
                break


def iter_hours(start_hour: datetime, end_hour: datetime):
    current = start_hour
    while current <= end_hour:
        yield current
        current += timedelta(hours=1)


def gharchive_url(base_url: str, hour: datetime) -> str:
    return f"{base_url}/{hour.strftime('%Y-%m-%d-%H')}.json.gz"


def iter_gharchive_events(base_url: str, start_hour: datetime, end_hour: datetime):
    for hour in iter_hours(start_hour, end_hour):
        url = gharchive_url(base_url, hour)
        request = Request(url, headers={"User-Agent": "IncidentNewsDatasetLoader/1.0"})
        try:
            with urlopen(request, timeout=120) as response:
                with gzip.GzipFile(fileobj=response) as archive:
                    for raw_line in archive:
                        yield json.loads(raw_line)
        except (HTTPError, URLError, TimeoutError) as exc:
            print(f"Skipped GH Archive hour {url}: {exc}")


def github_event_text(event: dict) -> str | None:
    event_type = event.get("type")
    payload = event.get("payload") or {}
    repo = event.get("repo", {}).get("name", "unknown/repo")

    if event_type == "PushEvent":
        commits = payload.get("commits") or []
        messages = [commit.get("message", "") for commit in commits[:5] if commit.get("message")]
        if not messages:
            return None
        return f"{repo} push: " + " | ".join(messages)

    if event_type == "IssuesEvent":
        issue = payload.get("issue") or {}
        return f"{repo} issue {payload.get('action')}: {issue.get('title', '')}\n{issue.get('body') or ''}"

    if event_type == "IssueCommentEvent":
        issue = payload.get("issue") or {}
        comment = payload.get("comment") or {}
        return f"{repo} issue comment: {issue.get('title', '')}\n{comment.get('body') or ''}"

    if event_type == "PullRequestEvent":
        pull_request = payload.get("pull_request") or {}
        return f"{repo} pull request {payload.get('action')}: {pull_request.get('title', '')}\n{pull_request.get('body') or ''}"

    if event_type == "PullRequestReviewEvent":
        review = payload.get("review") or {}
        pull_request = payload.get("pull_request") or {}
        return f"{repo} pull request review: {pull_request.get('title', '')}\n{review.get('body') or ''}"

    if event_type == "PullRequestReviewCommentEvent":
        comment = payload.get("comment") or {}
        pull_request = payload.get("pull_request") or {}
        return f"{repo} pull request review comment: {pull_request.get('title', '')}\n{comment.get('body') or ''}"

    if event_type == "ReleaseEvent":
        release = payload.get("release") or {}
        tag_name = release.get("tag_name") or release.get("name") or "release"
        return f"{repo} release {payload.get('action')}: {tag_name}\n{release.get('body') or ''}"

    return None


def github_release_payload(event: dict) -> tuple[str, str, str, datetime] | None:
    release = (event.get("payload") or {}).get("release") or {}
    repo = event.get("repo", {}).get("name", "unknown/repo")
    url = release.get("html_url")
    if not url:
        return None
    tag_name = release.get("tag_name") or release.get("name") or "release"
    title = f"{repo}: release {tag_name}"
    body = release.get("body") or ""
    published_at = parse_datetime(release.get("published_at")) or parse_datetime(event.get("created_at"))
    return title, body, url, published_at


def insert_gharchive_open_source(
    conn: psycopg.Connection,
    base_url: str,
    start_hour: datetime,
    end_hour: datetime,
    max_projects: int,
    log_limit: int,
    news_limit: int,
) -> tuple[int, int, int]:
    if max_projects <= 0 or (log_limit <= 0 and news_limit <= 0):
        return 0, 0, 0

    release_projects: dict[str, int] = {}
    all_projects: dict[str, int] = {}
    for event in iter_gharchive_events(base_url, start_hour, end_hour):
        repo = event.get("repo", {}).get("name")
        if not repo:
            continue
        all_projects[repo] = all_projects.get(repo, 0) + 1
        if event.get("type") == "ReleaseEvent":
            release_projects[repo] = release_projects.get(repo, 0) + 1

    selected = [
        repo
        for repo, _ in sorted(release_projects.items(), key=lambda item: item[1], reverse=True)
    ][:max_projects]
    if len(selected) < max_projects:
        for repo, _ in sorted(all_projects.items(), key=lambda item: item[1], reverse=True):
            if repo not in selected:
                selected.append(repo)
            if len(selected) >= max_projects:
                break
    selected_projects = set(selected)

    inserted_logs = 0
    inserted_news = 0
    with conn.cursor() as cur:
        for event in iter_gharchive_events(base_url, start_hour, end_hour):
            repo = event.get("repo", {}).get("name")
            if repo not in selected_projects:
                continue

            event_time = parse_datetime(event.get("created_at"))
            if event.get("type") == "ReleaseEvent" and inserted_news < news_limit:
                release_payload = github_release_payload(event)
                if release_payload is not None:
                    title, body, url, published_at = release_payload
                    cur.execute(
                        """
                        INSERT INTO raw_news (
                            source,
                            source_type,
                            url,
                            url_hash,
                            title,
                            body,
                            language,
                            published_at,
                            raw_payload,
                            content_hash
                        )
                        VALUES (%s, 'github_release', %s, %s, %s, %s, 'en', %s, %s::jsonb, %s)
                        ON CONFLICT (source, url_hash)
                        WHERE url_hash IS NOT NULL
                        DO UPDATE SET title = EXCLUDED.title,
                                      body = EXCLUDED.body,
                                      published_at = EXCLUDED.published_at,
                                      raw_payload = EXCLUDED.raw_payload,
                                      last_seen_at = now(),
                                      fetch_count = raw_news.fetch_count + 1
                        """,
                        (
                            repo,
                            url,
                            sha256_hex(url),
                            title,
                            body,
                            published_at,
                            json.dumps(
                                {
                                    "dataset": "gharchive",
                                    "repo": repo,
                                    "event_id": event.get("id"),
                                    "event_type": event.get("type"),
                                }
                            ),
                            sha256_hex(f"{title}\n{body}\n{url}"),
                        ),
                    )
                    inserted_news += 1

            if inserted_logs >= log_limit:
                continue

            message = github_event_text(event)
            if not message:
                continue
            cur.execute(
                """
                INSERT INTO raw_logs (
                    dataset,
                    source,
                    line_number,
                    message,
                    severity,
                    event_time,
                    raw_payload,
                    content_hash
                )
                VALUES ('gharchive_open_source', %s, %s, %s, %s, %s, %s::jsonb, %s)
                ON CONFLICT (dataset, source, line_number)
                DO UPDATE SET message = EXCLUDED.message,
                              severity = EXCLUDED.severity,
                              event_time = EXCLUDED.event_time,
                              raw_payload = EXCLUDED.raw_payload,
                              content_hash = EXCLUDED.content_hash
                """,
                (
                    repo,
                    stable_int_id(event.get("id", f"{repo}:{event_time}")),
                    message,
                    detect_severity(message),
                    event_time,
                    json.dumps(
                        {
                            "dataset": "gharchive",
                            "repo": repo,
                            "event_id": event.get("id"),
                            "event_type": event.get("type"),
                            "actor": (event.get("actor") or {}).get("login"),
                        }
                    ),
                    sha256_hex(message),
                ),
            )
            inserted_logs += 1

            if (inserted_logs + inserted_news) % 5_000 == 0:
                conn.commit()

    conn.commit()
    return len(selected_projects), inserted_logs, inserted_news


def insert_news(conn: psycopg.Connection, dataset_name: str, split: str, limit: int) -> int:
    if limit <= 0:
        return 0

    dataset = load_dataset(dataset_name, split=split, streaming=True)
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO news_sources (name, source_type, url, country, language, metadata)
            VALUES (%s, 'huggingface', %s, 'US', 'en', %s::jsonb)
            ON CONFLICT (name)
            DO UPDATE SET url = EXCLUDED.url, metadata = EXCLUDED.metadata, updated_at = now()
            RETURNING id
            """,
            (
                dataset_name,
                f"https://huggingface.co/datasets/{dataset_name}",
                json.dumps({"seeded": True, "loader": "data.import_datasets"}),
            ),
        )
        source_id = cur.fetchone()[0]

        inserted = 0
        base_time = datetime(2004, 8, 1, tzinfo=UTC)
        for index, row in enumerate(dataset):
            text = str(row.get("text") or "")
            if not text.strip():
                continue
            label = int(row.get("label", -1))
            title, body = news_title_and_body(text)
            url = f"hf://datasets/{dataset_name}/{split}/{index}"
            published_at = base_time + timedelta(minutes=index)
            content_hash = sha256_hex(f"{title}\n{body or ''}")
            cur.execute(
                """
                INSERT INTO raw_news (
                    source,
                    source_id,
                    source_type,
                    url,
                    url_hash,
                    title,
                    body,
                    language,
                    published_at,
                    raw_payload,
                    content_hash
                )
                VALUES (%s, %s, 'huggingface', %s, %s, %s, %s, 'en', %s, %s::jsonb, %s)
                ON CONFLICT (source, url_hash)
                WHERE url_hash IS NOT NULL
                DO UPDATE SET last_seen_at = now(), fetch_count = raw_news.fetch_count + 1
                """,
                (
                    dataset_name,
                    source_id,
                    url,
                    sha256_hex(url),
                    title,
                    body,
                    published_at,
                    json.dumps(
                        {
                            "dataset": dataset_name,
                            "split": split,
                            "row_index": index,
                            "label": label,
                            "label_name": NEWS_LABELS.get(label, "unknown"),
                            "note": "AG News has no original row timestamp; published_at is synthetic.",
                        }
                    ),
                    content_hash,
                ),
            )
            inserted += 1
            if inserted >= limit:
                break
    conn.commit()
    return inserted


def insert_logs(conn: psycopg.Connection, logs_url: str, limit: int) -> int:
    if limit <= 0:
        return 0

    inserted = 0
    with conn.cursor() as cur:
        for line_number, message in iter_remote_lines(logs_url, limit):
            cur.execute(
                """
                INSERT INTO raw_logs (
                    dataset,
                    source,
                    line_number,
                    message,
                    severity,
                    event_time,
                    raw_payload,
                    content_hash
                )
                VALUES ('logpai/loghub', 'Apache', %s, %s, %s, %s, %s::jsonb, %s)
                ON CONFLICT (dataset, source, line_number)
                DO UPDATE SET message = EXCLUDED.message,
                              severity = EXCLUDED.severity,
                              event_time = EXCLUDED.event_time,
                              raw_payload = EXCLUDED.raw_payload,
                              content_hash = EXCLUDED.content_hash
                """,
                (
                    line_number,
                    message,
                    detect_severity(message),
                    parse_apache_event_time(message),
                    json.dumps({"url": logs_url, "line_number": line_number}),
                    sha256_hex(message),
                ),
            )
            inserted += 1
    conn.commit()
    return inserted


def insert_hf_loghub_logs(conn: psycopg.Connection, dataset_name: str, split: str, limit: int) -> int:
    if limit <= 0:
        return 0

    dataset = load_dataset(dataset_name, split=split, streaming=True)
    inserted = 0
    batch = []
    batch_size = 5_000

    with conn.cursor() as cur:
        for index, row in enumerate(dataset, start=1):
            message = str(row.get("text") or row.get("message") or "").strip()
            if not message:
                continue

            batch.append(
                (
                    dataset_name,
                    "Loghub",
                    index,
                    message,
                    detect_severity(message),
                    parse_apache_event_time(message),
                    json.dumps({"dataset": dataset_name, "split": split, "row_index": index}),
                    sha256_hex(message),
                )
            )
            inserted += 1

            if len(batch) >= batch_size:
                cur.executemany(
                    """
                    INSERT INTO raw_logs (
                        dataset,
                        source,
                        line_number,
                        message,
                        severity,
                        event_time,
                        raw_payload,
                        content_hash
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s)
                    ON CONFLICT (dataset, source, line_number)
                    DO UPDATE SET message = EXCLUDED.message,
                                  severity = EXCLUDED.severity,
                                  event_time = EXCLUDED.event_time,
                                  raw_payload = EXCLUDED.raw_payload,
                                  content_hash = EXCLUDED.content_hash
                    """,
                    batch,
                )
                conn.commit()
                batch.clear()

            if inserted >= limit:
                break

        if batch:
            cur.executemany(
                """
                INSERT INTO raw_logs (
                    dataset,
                    source,
                    line_number,
                    message,
                    severity,
                    event_time,
                    raw_payload,
                    content_hash
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s)
                ON CONFLICT (dataset, source, line_number)
                DO UPDATE SET message = EXCLUDED.message,
                              severity = EXCLUDED.severity,
                              event_time = EXCLUDED.event_time,
                              raw_payload = EXCLUDED.raw_payload,
                              content_hash = EXCLUDED.content_hash
                """,
                batch,
            )
            conn.commit()

    return inserted


def fetch_json(url: str) -> dict:
    request = Request(url, headers={"User-Agent": "IncidentNewsDatasetLoader/1.0"})
    with urlopen(request, timeout=60) as response:
        return json.load(response)


def fetch_hn_stories(search_url: str, query: str, hits_per_page: int) -> list[dict]:
    params = urlencode({"query": query, "tags": "story", "hitsPerPage": hits_per_page})
    return fetch_json(f"{search_url}?{params}").get("hits", [])


def insert_hackernews_provider_news(
    conn: psycopg.Connection,
    search_url: str,
    limit_per_provider: int,
) -> int:
    if limit_per_provider <= 0:
        return 0

    inserted = 0
    with conn.cursor() as cur:
        for provider in STATUSPAGE_SOURCES:
            seen_ids: set[str] = set()
            queries = [
                f"{provider} outage",
                f"{provider} incident",
                f"{provider} status",
            ]
            for query in queries:
                remaining = limit_per_provider - len(seen_ids)
                if remaining <= 0:
                    break
                try:
                    stories = fetch_hn_stories(search_url, query, min(remaining, 100))
                except Exception as exc:
                    print(f"Skipped Hacker News query {query!r}: {exc}")
                    continue

                for story in stories:
                    object_id = str(story.get("objectID") or "")
                    if not object_id or object_id in seen_ids:
                        continue
                    seen_ids.add(object_id)
                    title = story.get("title") or story.get("story_title") or "Untitled Hacker News story"
                    body = story.get("story_text") or story.get("comment_text") or ""
                    url = story.get("url") or f"https://news.ycombinator.com/item?id={object_id}"
                    published_at = parse_datetime(story.get("created_at"))
                    cur.execute(
                        """
                        INSERT INTO raw_news (
                            source,
                            source_type,
                            url,
                            url_hash,
                            title,
                            body,
                            language,
                            published_at,
                            raw_payload,
                            content_hash
                        )
                        VALUES ('hackernews_algolia', 'hackernews_story', %s, %s, %s, %s, 'en', %s, %s::jsonb, %s)
                        ON CONFLICT (source, url_hash)
                        WHERE url_hash IS NOT NULL
                        DO UPDATE SET title = EXCLUDED.title,
                                      body = EXCLUDED.body,
                                      published_at = EXCLUDED.published_at,
                                      raw_payload = EXCLUDED.raw_payload,
                                      last_seen_at = now(),
                                      fetch_count = raw_news.fetch_count + 1
                        """,
                        (
                            url,
                            sha256_hex(f"hn:{object_id}"),
                            title,
                            body,
                            published_at,
                            json.dumps(
                                {
                                    "dataset": "hackernews_algolia",
                                    "provider": provider,
                                    "query": query,
                                    "object_id": object_id,
                                    "author": story.get("author"),
                                    "points": story.get("points"),
                                    "num_comments": story.get("num_comments"),
                                    "story_url": url,
                                }
                            ),
                            sha256_hex(f"{title}\n{body}\n{url}"),
                        ),
                    )
                    inserted += 1
            conn.commit()

    return inserted


def parse_rss_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    parsed = parsedate_to_datetime(value)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def fetch_google_news_entries(rss_url: str, query: str) -> list[dict]:
    params = urlencode({"q": query, "hl": "en-US", "gl": "US", "ceid": "US:en"})
    request = Request(f"{rss_url}?{params}", headers={"User-Agent": "IncidentNewsDatasetLoader/1.0"})
    with urlopen(request, timeout=60) as response:
        feed = feedparser.parse(response.read())
    return list(feed.entries)


def insert_google_news_provider_news(
    conn: psycopg.Connection,
    rss_url: str,
    limit_per_provider: int,
) -> int:
    if limit_per_provider <= 0:
        return 0

    inserted = 0
    with conn.cursor() as cur:
        for provider in STATUSPAGE_SOURCES:
            seen_links: set[str] = set()
            queries = [
                f"{provider} outage when:90d",
                f"{provider} down when:90d",
                f"{provider} incident when:90d",
                f"{provider} status when:90d",
            ]
            for query in queries:
                remaining = limit_per_provider - len(seen_links)
                if remaining <= 0:
                    break
                try:
                    entries = fetch_google_news_entries(rss_url, query)
                except Exception as exc:
                    print(f"Skipped Google News query {query!r}: {exc}")
                    continue

                for entry in entries[:remaining]:
                    link = entry.get("link")
                    if not link or link in seen_links:
                        continue
                    seen_links.add(link)
                    title = entry.get("title") or "Untitled Google News story"
                    published_at = parse_rss_datetime(entry.get("published"))
                    cur.execute(
                        """
                        INSERT INTO raw_news (
                            source,
                            source_type,
                            url,
                            url_hash,
                            title,
                            body,
                            language,
                            published_at,
                            raw_payload,
                            content_hash
                        )
                        VALUES ('google_news_rss', 'google_news_story', %s, %s, %s, %s, 'en', %s, %s::jsonb, %s)
                        ON CONFLICT (source, url_hash)
                        WHERE url_hash IS NOT NULL
                        DO UPDATE SET title = EXCLUDED.title,
                                      body = EXCLUDED.body,
                                      published_at = EXCLUDED.published_at,
                                      raw_payload = EXCLUDED.raw_payload,
                                      last_seen_at = now(),
                                      fetch_count = raw_news.fetch_count + 1
                        """,
                        (
                            link,
                            sha256_hex(link),
                            title,
                            entry.get("summary") or "",
                            published_at,
                            json.dumps(
                                {
                                    "dataset": "google_news_rss",
                                    "provider": provider,
                                    "query": query,
                                    "feed_id": entry.get("id"),
                                    "published": entry.get("published"),
                                    "source_title": entry.get("source", {}).get("title")
                                    if isinstance(entry.get("source"), dict)
                                    else None,
                                }
                            ),
                            sha256_hex(f"{title}\n{link}"),
                        ),
                    )
                    inserted += 1
            conn.commit()

    return inserted


def query_osv_package(query_url: str, ecosystem: str, package_name: str) -> list[dict]:
    request_body = json.dumps(
        {"package": {"ecosystem": ecosystem, "name": package_name}}
    ).encode("utf-8")
    request = Request(
        query_url,
        data=request_body,
        headers={
            "Content-Type": "application/json",
            "User-Agent": "IncidentNewsDatasetLoader/1.0",
        },
    )
    with urlopen(request, timeout=60) as response:
        return json.load(response).get("vulns", [])


def advisory_references(vuln: dict) -> list[str]:
    refs = vuln.get("references") or []
    return [ref.get("url") for ref in refs if ref.get("url")]


def insert_osv_advisories(conn: psycopg.Connection, query_url: str, limit_per_package: int) -> tuple[int, int]:
    if limit_per_package <= 0:
        return 0, 0

    inserted_news = 0
    inserted_logs = 0
    with conn.cursor() as cur:
        for ecosystem, package_name in OSV_PACKAGES:
            try:
                vulns = query_osv_package(query_url, ecosystem, package_name)
            except Exception as exc:
                print(f"Skipped OSV package {ecosystem}/{package_name}: {exc}")
                continue

            for vuln in vulns[:limit_per_package]:
                advisory_id = vuln.get("id")
                if not advisory_id:
                    continue
                published_at = parse_datetime(vuln.get("published")) or parse_datetime(vuln.get("modified"))
                title = vuln.get("summary") or f"{ecosystem}/{package_name} advisory {advisory_id}"
                details = vuln.get("details") or ""
                references = advisory_references(vuln)
                url = references[0] if references else f"https://osv.dev/vulnerability/{advisory_id}"
                aliases = vuln.get("aliases") or []

                cur.execute(
                    """
                    INSERT INTO raw_news (
                        source,
                        source_type,
                        url,
                        url_hash,
                        title,
                        body,
                        language,
                        published_at,
                        raw_payload,
                        content_hash
                    )
                    VALUES ('osv.dev', 'osv_advisory', %s, %s, %s, %s, 'en', %s, %s::jsonb, %s)
                    ON CONFLICT (source, url_hash)
                    WHERE url_hash IS NOT NULL
                    DO UPDATE SET title = EXCLUDED.title,
                                  body = EXCLUDED.body,
                                  published_at = EXCLUDED.published_at,
                                  raw_payload = EXCLUDED.raw_payload,
                                  last_seen_at = now(),
                                  fetch_count = raw_news.fetch_count + 1
                    """,
                    (
                        url,
                        sha256_hex(f"osv:{advisory_id}"),
                        title,
                        details,
                        published_at,
                        json.dumps(
                            {
                                "dataset": "osv.dev",
                                "advisory_id": advisory_id,
                                "aliases": aliases,
                                "ecosystem": ecosystem,
                                "package": package_name,
                                "modified": vuln.get("modified"),
                                "published": vuln.get("published"),
                                "references": references,
                            }
                        ),
                        sha256_hex(f"{advisory_id}\n{title}\n{details}"),
                    ),
                )
                inserted_news += 1

                message = f"{ecosystem}/{package_name} affected by {advisory_id}: {title}"
                cur.execute(
                    """
                    INSERT INTO raw_logs (
                        dataset,
                        source,
                        line_number,
                        message,
                        severity,
                        event_time,
                        raw_payload,
                        content_hash
                    )
                    VALUES ('osv_advisories', %s, %s, %s, 'warn', %s, %s::jsonb, %s)
                    ON CONFLICT (dataset, source, line_number)
                    DO UPDATE SET message = EXCLUDED.message,
                                  severity = EXCLUDED.severity,
                                  event_time = EXCLUDED.event_time,
                                  raw_payload = EXCLUDED.raw_payload,
                                  content_hash = EXCLUDED.content_hash
                    """,
                    (
                        f"{ecosystem}/{package_name}",
                        stable_int_id(f"{ecosystem}:{package_name}:{advisory_id}"),
                        message,
                        published_at,
                        json.dumps(
                            {
                                "dataset": "osv.dev",
                                "advisory_id": advisory_id,
                                "ecosystem": ecosystem,
                                "package": package_name,
                                "aliases": aliases,
                            }
                        ),
                        sha256_hex(message),
                    ),
                )
                inserted_logs += 1

            conn.commit()

    return inserted_news, inserted_logs


def statuspage_severity(impact: str | None) -> str:
    return {
        "critical": "error",
        "major": "error",
        "minor": "warn",
        "maintenance": "info",
        "none": "info",
    }.get((impact or "").lower(), "info")


def affected_component_names(update: dict) -> list[str]:
    components = update.get("affected_components") or []
    return [component.get("name") for component in components if component.get("name")]


def insert_statuspage_incidents(conn: psycopg.Connection, limit: int) -> tuple[int, int]:
    if limit <= 0:
        return 0, 0

    inserted_news = 0
    inserted_logs = 0
    per_source_limit = max(1, limit // len(STATUSPAGE_SOURCES))

    with conn.cursor() as cur:
        for source_name, api_url in STATUSPAGE_SOURCES.items():
            payload = fetch_json(api_url)
            page_url = payload.get("page", {}).get("url") or api_url
            cur.execute(
                """
                INSERT INTO news_sources (name, source_type, url, country, language, metadata)
                VALUES (%s, 'statuspage', %s, 'GLOBAL', 'en', %s::jsonb)
                ON CONFLICT (name)
                DO UPDATE SET url = EXCLUDED.url, metadata = EXCLUDED.metadata, updated_at = now()
                RETURNING id
                """,
                (
                    f"{source_name} Statuspage",
                    page_url,
                    json.dumps({"seeded": True, "loader": "data.import_datasets", "api_url": api_url}),
                ),
            )
            source_id = cur.fetchone()[0]

            used_for_source = 0
            for incident in payload.get("incidents", []):
                if used_for_source >= per_source_limit:
                    break

                incident_id = incident["id"]
                started_at = parse_datetime(incident.get("started_at") or incident.get("created_at"))
                resolved_at = parse_datetime(incident.get("resolved_at"))
                updates = sorted(
                    incident.get("incident_updates", []),
                    key=lambda item: item.get("display_at") or item.get("created_at") or "",
                )
                body = "\n".join(
                    update.get("body", "")
                    for update in updates
                    if update.get("body")
                )
                components = sorted(
                    {
                        name
                        for update in updates
                        for name in affected_component_names(update)
                    }
                )
                title = f"{source_name}: {incident.get('name', 'status incident')}"
                row_url = incident.get("shortlink") or f"{page_url}/incidents/{incident_id}"
                cur.execute(
                    """
                    INSERT INTO raw_news (
                        source,
                        source_id,
                        source_type,
                        url,
                        url_hash,
                        title,
                        body,
                        language,
                        published_at,
                        raw_payload,
                        content_hash
                    )
                    VALUES (%s, %s, 'statuspage_incident', %s, %s, %s, %s, 'en', %s, %s::jsonb, %s)
                    ON CONFLICT (source, url_hash)
                    WHERE url_hash IS NOT NULL
                    DO UPDATE SET title = EXCLUDED.title,
                                  body = EXCLUDED.body,
                                  published_at = EXCLUDED.published_at,
                                  raw_payload = EXCLUDED.raw_payload,
                                  last_seen_at = now(),
                                  fetch_count = raw_news.fetch_count + 1
                    """,
                    (
                        f"{source_name} Statuspage",
                        source_id,
                        row_url,
                        sha256_hex(row_url),
                        title,
                        body,
                        started_at,
                        json.dumps(
                            {
                                "dataset": "statuspage_incidents",
                                "provider": source_name,
                                "incident_id": incident_id,
                                "status": incident.get("status"),
                                "impact": incident.get("impact"),
                                "started_at": incident.get("started_at"),
                                "resolved_at": incident.get("resolved_at"),
                                "components": components,
                            }
                        ),
                        sha256_hex(f"{title}\n{body}"),
                    ),
                )
                inserted_news += 1

                for update_index, update in enumerate(updates, start=1):
                    message = f"{source_name} {incident.get('name', 'incident')} [{update.get('status')}]: {update.get('body', '')}"
                    event_time = parse_datetime(update.get("display_at") or update.get("created_at"))
                    cur.execute(
                        """
                        INSERT INTO raw_logs (
                            dataset,
                            source,
                            line_number,
                            message,
                            severity,
                            event_time,
                            raw_payload,
                            content_hash
                        )
                        VALUES ('statuspage_incidents', %s, %s, %s, %s, %s, %s::jsonb, %s)
                        ON CONFLICT (dataset, source, line_number)
                        DO UPDATE SET message = EXCLUDED.message,
                                      severity = EXCLUDED.severity,
                                      event_time = EXCLUDED.event_time,
                                      raw_payload = EXCLUDED.raw_payload,
                                      content_hash = EXCLUDED.content_hash
                        """,
                        (
                            source_name,
                            stable_int_id(f"{source_name}:{incident_id}:{update.get('id') or update_index}"),
                            message,
                            statuspage_severity(incident.get("impact")),
                            event_time,
                            json.dumps(
                                {
                                    "provider": source_name,
                                    "incident_id": incident_id,
                                    "incident_name": incident.get("name"),
                                    "update_id": update.get("id"),
                                    "update_status": update.get("status"),
                                    "impact": incident.get("impact"),
                                    "components": affected_component_names(update),
                                    "linked_news_url": row_url,
                                }
                            ),
                            sha256_hex(message),
                        ),
                    )
                    inserted_logs += 1

                used_for_source += 1
                if inserted_news >= limit:
                    conn.commit()
                    return inserted_news, inserted_logs

    conn.commit()
    return inserted_news, inserted_logs


def ensure_gdelt_cache(url: str, cache_path: Path) -> Path:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if not cache_path.exists():
        urlretrieve(url, cache_path)
    return cache_path


def gdelt_title(row: list[str]) -> str:
    actor1 = row[6] or row[5] or "Unknown actor"
    actor2 = row[16] or row[15] or "unknown target"
    event_code = row[26] or "unknown"
    return f"GDELT event {event_code}: {actor1} -> {actor2}"


def gdelt_body(row: list[str]) -> str:
    parts = [
        f"Actor1: {row[6] or row[5] or 'unknown'}",
        f"Actor2: {row[16] or row[15] or 'unknown'}",
        f"EventCode: {row[26] or 'unknown'}",
        f"EventRootCode: {row[28] or 'unknown'}",
        f"QuadClass: {row[29] or 'unknown'}",
        f"GoldsteinScale: {row[30] or 'unknown'}",
        f"NumMentions: {row[31] or 'unknown'}",
        f"NumSources: {row[32] or 'unknown'}",
        f"NumArticles: {row[33] or 'unknown'}",
        f"AvgTone: {row[34] or 'unknown'}",
    ]
    return "; ".join(parts)


def insert_gdelt_events(
    conn: psycopg.Connection,
    url: str,
    cache_path: Path,
    start_date: datetime,
    end_date: datetime,
    limit: int,
) -> int:
    if limit <= 0:
        return 0

    archive_path = ensure_gdelt_cache(url, cache_path)
    start_key = int(start_date.strftime("%Y%m%d"))
    end_key = int(end_date.strftime("%Y%m%d"))
    inserted = 0

    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO news_sources (name, source_type, url, country, language, metadata)
            VALUES (%s, 'gdelt_event', %s, 'GLOBAL', 'en', %s::jsonb)
            ON CONFLICT (name)
            DO UPDATE SET url = EXCLUDED.url, metadata = EXCLUDED.metadata, updated_at = now()
            RETURNING id
            """,
            (
                GDELT_2005_SOURCE_NAME,
                url,
                json.dumps({"seeded": True, "loader": "data.import_datasets"}),
            ),
        )
        source_id = cur.fetchone()[0]

        with zipfile.ZipFile(archive_path) as archive:
            with archive.open(archive.namelist()[0]) as raw_file:
                text_file = (line.decode("latin-1") for line in raw_file)
                reader = csv.reader(text_file, delimiter="\t")
                for row in reader:
                    if len(row) < 57:
                        continue

                    sql_date = int(row[1])
                    if sql_date < start_key:
                        continue
                    if sql_date > end_key:
                        break

                    event_id = row[0]
                    published_at = datetime.strptime(row[1], "%Y%m%d").replace(tzinfo=UTC)
                    title = gdelt_title(row)
                    body = gdelt_body(row)
                    row_url = f"gdelt://events/2005/{event_id}"
                    cur.execute(
                        """
                        INSERT INTO raw_news (
                            source,
                            source_id,
                            source_type,
                            url,
                            url_hash,
                            title,
                            body,
                            language,
                            published_at,
                            raw_payload,
                            content_hash
                        )
                        VALUES (%s, %s, 'gdelt_event', %s, %s, %s, %s, 'en', %s, %s::jsonb, %s)
                        ON CONFLICT (source, url_hash)
                        WHERE url_hash IS NOT NULL
                        DO UPDATE SET last_seen_at = now(), fetch_count = raw_news.fetch_count + 1
                        """,
                        (
                            GDELT_2005_SOURCE_NAME,
                            source_id,
                            row_url,
                            sha256_hex(row_url),
                            title,
                            body,
                            published_at,
                            json.dumps(
                                {
                                    "dataset": "GDELT 1.0 reduced events",
                                    "global_event_id": event_id,
                                    "sql_date": row[1],
                                    "actor1": row[6],
                                    "actor2": row[16],
                                    "event_code": row[26],
                                    "event_root_code": row[28],
                                    "quad_class": row[29],
                                    "goldstein_scale": row[30],
                                    "num_mentions": row[31],
                                    "num_sources": row[32],
                                    "num_articles": row[33],
                                    "avg_tone": row[34],
                                }
                            ),
                            sha256_hex(f"{title}\n{body}"),
                        ),
                    )
                    inserted += 1
                    if inserted >= limit:
                        conn.commit()
                        return inserted

    conn.commit()
    return inserted


def gdeltv2_export_urls(
    master_url: str,
    max_files: int,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
    max_files_per_day: int | None = None,
) -> list[str]:
    content = urlopen(master_url, timeout=60).read().decode("utf-8")
    urls_by_day: dict[str, list[str]] = {}
    all_urls = []
    for line in content.splitlines():
        if not line.endswith(".export.CSV.zip"):
            continue
        url = line.split()[-1]
        filename = Path(url).name
        timestamp = filename.split(".")[0]
        if start_date is not None and timestamp[:8] < start_date.strftime("%Y%m%d"):
            continue
        if end_date is not None and timestamp[:8] > end_date.strftime("%Y%m%d"):
            continue
        all_urls.append(url)
        urls_by_day.setdefault(timestamp[:8], []).append(url)

    if start_date is None and end_date is None:
        return list(reversed(all_urls[-max_files:]))

    selected = []
    for day in sorted(urls_by_day):
        day_urls = urls_by_day[day]
        if max_files_per_day is not None:
            step = max(1, len(day_urls) // max_files_per_day)
            day_urls = day_urls[::step][:max_files_per_day]
        selected.extend(day_urls)
    return selected[:max_files]


def gdeltv2_cache_path(url: str) -> Path:
    return Path("data/cache/gdeltv2") / Path(url).name


def insert_gdeltv2_events(
    conn: psycopg.Connection,
    master_url: str,
    limit: int,
    max_files: int,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
    max_files_per_day: int | None = None,
) -> int:
    if limit <= 0:
        return 0

    inserted = 0
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO news_sources (name, source_type, url, country, language, metadata)
            VALUES (%s, 'gdeltv2_event', %s, 'GLOBAL', 'en', %s::jsonb)
            ON CONFLICT (name)
            DO UPDATE SET url = EXCLUDED.url, metadata = EXCLUDED.metadata, updated_at = now()
            RETURNING id
            """,
            (
                GDELTV2_SOURCE_NAME,
                master_url,
                json.dumps({"seeded": True, "loader": "data.import_datasets", "source": "GDELT 2.1 Events"}),
            ),
        )
        source_id = cur.fetchone()[0]

        for url in gdeltv2_export_urls(master_url, max_files, start_date, end_date, max_files_per_day):
            archive_path = gdeltv2_cache_path(url)
            ensure_gdelt_cache(url, archive_path)

            with zipfile.ZipFile(archive_path) as archive:
                with archive.open(archive.namelist()[0]) as raw_file:
                    text_file = (line.decode("latin-1") for line in raw_file)
                    reader = csv.reader(text_file, delimiter="\t")
                    for row in reader:
                        if len(row) < 61:
                            continue

                        event_id = row[0]
                        date_added = datetime.strptime(row[59], "%Y%m%d%H%M%S").replace(tzinfo=UTC)
                        source_url = row[60] or f"gdeltv2://events/{event_id}"
                        row_url = f"gdeltv2://events/{event_id}"
                        title = gdelt_title(row)
                        body = gdelt_body(row)
                        cur.execute(
                            """
                            INSERT INTO raw_news (
                                source,
                                source_id,
                                source_type,
                                url,
                                url_hash,
                                title,
                                body,
                                language,
                                published_at,
                                raw_payload,
                                content_hash
                            )
                            VALUES (%s, %s, 'gdeltv2_event', %s, %s, %s, %s, 'en', %s, %s::jsonb, %s)
                            ON CONFLICT (source, url_hash)
                            WHERE url_hash IS NOT NULL
                            DO UPDATE SET title = EXCLUDED.title,
                                          body = EXCLUDED.body,
                                          published_at = EXCLUDED.published_at,
                                          raw_payload = EXCLUDED.raw_payload,
                                          last_seen_at = now(),
                                          fetch_count = raw_news.fetch_count + 1
                            """,
                            (
                                GDELTV2_SOURCE_NAME,
                                source_id,
                                row_url,
                                sha256_hex(row_url),
                                title,
                                body,
                                date_added,
                                json.dumps(
                                    {
                                        "dataset": "GDELT 2.1 Events",
                                        "global_event_id": event_id,
                                        "sql_date": row[1],
                                        "date_added": row[59],
                                        "actor1": row[6],
                                        "actor2": row[16],
                                        "event_code": row[26],
                                        "event_root_code": row[28],
                                        "quad_class": row[29],
                                        "goldstein_scale": row[30],
                                        "num_mentions": row[31],
                                        "num_sources": row[32],
                                        "num_articles": row[33],
                                        "avg_tone": row[34],
                                        "source_url": source_url,
                                    }
                                ),
                                sha256_hex(f"{title}\n{body}\n{source_url}"),
                            ),
                        )
                        inserted += 1
                        if inserted >= limit:
                            conn.commit()
                            return inserted
            conn.commit()

    return inserted


def count_objects(conn: psycopg.Connection) -> dict[str, int]:
    result: dict[str, int] = {}
    with conn.cursor() as cur:
        for table in ("raw_news", "raw_logs"):
            cur.execute(f"SELECT count(*) FROM {table}")
            result[table] = int(cur.fetchone()[0])
    result["total_selected"] = sum(result.values())
    return result


def main(args: argparse.Namespace) -> None:
    minimum_total = apply_profile(args)
    apply_migrations()
    with psycopg.connect(_postgres_url(settings.database_url)) as conn:
        news_count = insert_news(conn, args.news_dataset, args.news_split, args.news_limit)
        log_count = insert_logs(conn, args.logs_url, args.logs_limit)
        hf_loghub_count = insert_hf_loghub_logs(
            conn,
            args.hf_loghub_dataset,
            args.hf_loghub_split,
            args.hf_loghub_limit,
        )
        statuspage_news_count, statuspage_log_count = insert_statuspage_incidents(conn, args.statuspage_limit)
        hn_news_count = insert_hackernews_provider_news(
            conn,
            args.hn_search_url,
            args.hn_limit_per_provider,
        )
        google_news_count = insert_google_news_provider_news(
            conn,
            args.google_news_rss_url,
            args.google_news_limit_per_provider,
        )
        osv_news_count, osv_log_count = insert_osv_advisories(
            conn,
            args.osv_query_url,
            args.osv_limit_per_package,
        )
        gdelt_count = insert_gdelt_events(
            conn,
            args.gdelt_url,
            Path(args.gdelt_cache),
            args.gdelt_start,
            args.gdelt_end,
            args.gdelt_limit,
        )
        gdeltv2_count = insert_gdeltv2_events(
            conn,
            args.gdeltv2_master_url,
            args.gdeltv2_limit,
            args.gdeltv2_max_files,
            args.gdeltv2_start,
            args.gdeltv2_end,
            args.gdeltv2_max_files_per_day,
        )
        gharchive_projects, gharchive_logs, gharchive_news = insert_gharchive_open_source(
            conn,
            args.gharchive_base_url,
            args.gharchive_start,
            args.gharchive_end,
            args.gharchive_projects,
            args.gharchive_log_limit,
            args.gharchive_news_limit,
        )
        counts = count_objects(conn)

    print(f"Imported AG News rows: {news_count}")
    print(f"Imported Apache log rows: {log_count}")
    print(f"Imported HuggingFace Loghub rows: {hf_loghub_count}")
    print(f"Imported Statuspage incident rows: {statuspage_news_count}")
    print(f"Imported Statuspage update log rows: {statuspage_log_count}")
    print(f"Imported Hacker News story rows: {hn_news_count}")
    print(f"Imported Google News story rows: {google_news_count}")
    print(f"Imported OSV advisory rows: {osv_news_count}")
    print(f"Imported OSV package log rows: {osv_log_count}")
    print(f"Imported GDELT event rows: {gdelt_count}")
    print(f"Imported GDELT v2 event rows: {gdeltv2_count}")
    print(f"Imported GH Archive projects: {gharchive_projects}")
    print(f"Imported GH Archive log rows: {gharchive_logs}")
    print(f"Imported GH Archive release rows: {gharchive_news}")
    for table_name, value in counts.items():
        print(f"{table_name}: {value}")
    if counts["total_selected"] < minimum_total:
        raise SystemExit(f"Imported fewer than {minimum_total} selected objects.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import open-source news, event, and log datasets into PostgreSQL.")
    parser.add_argument("--profile", choices=("default", *PROFILE_PRESETS.keys()), default="default")
    parser.add_argument("--minimum-total", type=int, default=PROFILE_DEFAULT_MINIMUM_TOTAL)
    parser.add_argument("--news-dataset", default=DEFAULT_NEWS_DATASET)
    parser.add_argument("--news-split", default="train")
    parser.add_argument("--news-limit", type=int, default=50_000)
    parser.add_argument("--hf-loghub-dataset", default=DEFAULT_HF_LOGHUB_DATASET)
    parser.add_argument("--hf-loghub-split", default="train")
    parser.add_argument("--hf-loghub-limit", type=int, default=0)
    parser.add_argument("--logs-url", default=DEFAULT_LOG_URL)
    parser.add_argument("--logs-limit", type=int, default=2_000)
    parser.add_argument("--statuspage-limit", type=int, default=100)
    parser.add_argument("--hn-search-url", default=DEFAULT_HN_SEARCH_URL)
    parser.add_argument("--hn-limit-per-provider", type=int, default=0)
    parser.add_argument("--google-news-rss-url", default=DEFAULT_GOOGLE_NEWS_RSS_URL)
    parser.add_argument("--google-news-limit-per-provider", type=int, default=0)
    parser.add_argument("--osv-query-url", default=DEFAULT_OSV_QUERY_URL)
    parser.add_argument("--osv-limit-per-package", type=int, default=0)
    parser.add_argument("--gdelt-url", default=DEFAULT_GDELT_URL)
    parser.add_argument("--gdelt-cache", default=str(DEFAULT_GDELT_CACHE))
    parser.add_argument("--gdelt-start", type=parse_date, default=parse_date("2005-12-03"))
    parser.add_argument("--gdelt-end", type=parse_date, default=parse_date("2005-12-05"))
    parser.add_argument("--gdelt-limit", type=int, default=0)
    parser.add_argument("--gdeltv2-master-url", default=DEFAULT_GDELT_V2_MASTER_URL)
    parser.add_argument("--gdeltv2-limit", type=int, default=50_000)
    parser.add_argument("--gdeltv2-max-files", type=int, default=96)
    parser.add_argument("--gdeltv2-start", type=parse_date, default=None)
    parser.add_argument("--gdeltv2-end", type=parse_date, default=None)
    parser.add_argument("--gdeltv2-max-files-per-day", type=int, default=None)
    parser.add_argument("--gharchive-base-url", default=DEFAULT_GHARCHIVE_BASE_URL)
    parser.add_argument("--gharchive-start", type=parse_hour, default=parse_hour("2026-07-06T00"))
    parser.add_argument("--gharchive-end", type=parse_hour, default=parse_hour("2026-07-06T23"))
    parser.add_argument("--gharchive-projects", type=int, default=0)
    parser.add_argument("--gharchive-log-limit", type=int, default=0)
    parser.add_argument("--gharchive-news-limit", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
