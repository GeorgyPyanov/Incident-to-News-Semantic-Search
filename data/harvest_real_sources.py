import argparse
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import psycopg

from data.import_datasets import (
    DEFAULT_GDELT_V2_MASTER_URL,
    DEFAULT_GHARCHIVE_BASE_URL,
    DEFAULT_GOOGLE_NEWS_RSS_URL,
    DEFAULT_HN_SEARCH_URL,
    DEFAULT_OSV_QUERY_URL,
    count_objects,
    insert_gharchive_open_source,
    insert_gdeltv2_events,
    insert_google_news_provider_news,
    insert_hackernews_provider_news,
    insert_osv_advisories,
    insert_statuspage_incidents,
)
from database.migrate import _postgres_url, apply_migrations
from database.settings import settings


def log(message: str, log_path: Path) -> None:
    timestamp = datetime.now(UTC).isoformat(timespec="seconds")
    line = f"[{timestamp}] {message}"
    print(line, flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as file:
        file.write(line + "\n")


def run_cycle(conn: psycopg.Connection, args: argparse.Namespace, cycle: int, log_path: Path) -> None:
    now = datetime.now(UTC).replace(minute=0, second=0, microsecond=0)
    gh_start = now - timedelta(hours=args.gharchive_lookback_hours)
    gh_end = now - timedelta(hours=1)

    log(f"cycle={cycle} started", log_path)

    status_news, status_logs = insert_statuspage_incidents(conn, args.statuspage_limit)
    log(f"cycle={cycle} statuspage incidents={status_news} updates={status_logs}", log_path)

    hn_news = insert_hackernews_provider_news(conn, args.hn_search_url, args.hn_limit_per_provider)
    log(f"cycle={cycle} hackernews stories={hn_news}", log_path)

    google_news = insert_google_news_provider_news(
        conn,
        args.google_news_rss_url,
        args.google_news_limit_per_provider,
    )
    log(f"cycle={cycle} google_news stories={google_news}", log_path)

    osv_news = 0
    osv_logs = 0
    if args.osv_limit_per_package > 0 and cycle % args.osv_every_cycles == 0:
        osv_news, osv_logs = insert_osv_advisories(conn, args.osv_query_url, args.osv_limit_per_package)
    log(f"cycle={cycle} osv advisories={osv_news} package_logs={osv_logs}", log_path)

    gdelt_rows = insert_gdeltv2_events(
        conn,
        args.gdeltv2_master_url,
        args.gdeltv2_limit,
        args.gdeltv2_max_files,
        args.gdeltv2_start,
        args.gdeltv2_end,
        args.gdeltv2_max_files_per_day,
    )
    log(f"cycle={cycle} gdeltv2 rows={gdelt_rows}", log_path)

    gh_projects, gh_logs, gh_news = insert_gharchive_open_source(
        conn,
        args.gharchive_base_url,
        gh_start,
        gh_end,
        args.gharchive_projects,
        args.gharchive_log_limit,
        args.gharchive_news_limit,
    )
    log(
        f"cycle={cycle} gharchive projects={gh_projects} logs={gh_logs} releases={gh_news} "
        f"window={gh_start.isoformat()}..{gh_end.isoformat()}",
        log_path,
    )

    counts = count_objects(conn)
    log(f"cycle={cycle} counts={counts}", log_path)


def main(args: argparse.Namespace) -> None:
    apply_migrations()
    log_path = Path(args.log_path)
    deadline = time.monotonic() + args.duration_hours * 3600
    cycle = 1

    with psycopg.connect(_postgres_url(settings.database_url)) as conn:
        while True:
            try:
                run_cycle(conn, args, cycle, log_path)
            except Exception as exc:
                log(f"cycle={cycle} failed: {type(exc).__name__}: {exc}", log_path)

            cycle += 1
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            sleep_seconds = min(args.interval_minutes * 60, remaining)
            log(f"sleeping {int(sleep_seconds)} seconds", log_path)
            time.sleep(sleep_seconds)

    log("harvest finished", log_path)


def parse_date(value: str | None):
    if value is None:
        return None
    return datetime.fromisoformat(value).replace(tzinfo=UTC)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Continuously harvest real paired tech data sources.")
    parser.add_argument("--duration-hours", type=float, default=6.0)
    parser.add_argument("--interval-minutes", type=float, default=30.0)
    parser.add_argument("--log-path", default="logs/harvest_real_sources.log")
    parser.add_argument("--statuspage-limit", type=int, default=5000)
    parser.add_argument("--hn-search-url", default=DEFAULT_HN_SEARCH_URL)
    parser.add_argument("--hn-limit-per-provider", type=int, default=100)
    parser.add_argument("--google-news-rss-url", default=DEFAULT_GOOGLE_NEWS_RSS_URL)
    parser.add_argument("--google-news-limit-per-provider", type=int, default=80)
    parser.add_argument("--osv-query-url", default=DEFAULT_OSV_QUERY_URL)
    parser.add_argument("--osv-limit-per-package", type=int, default=20)
    parser.add_argument("--osv-every-cycles", type=int, default=6)
    parser.add_argument("--gdeltv2-master-url", default=DEFAULT_GDELT_V2_MASTER_URL)
    parser.add_argument("--gdeltv2-limit", type=int, default=10000)
    parser.add_argument("--gdeltv2-max-files", type=int, default=24)
    parser.add_argument("--gdeltv2-start", type=parse_date, default=None)
    parser.add_argument("--gdeltv2-end", type=parse_date, default=None)
    parser.add_argument("--gdeltv2-max-files-per-day", type=int, default=None)
    parser.add_argument("--gharchive-base-url", default=DEFAULT_GHARCHIVE_BASE_URL)
    parser.add_argument("--gharchive-lookback-hours", type=int, default=6)
    parser.add_argument("--gharchive-projects", type=int, default=300)
    parser.add_argument("--gharchive-log-limit", type=int, default=50000)
    parser.add_argument("--gharchive-news-limit", type=int, default=2000)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
