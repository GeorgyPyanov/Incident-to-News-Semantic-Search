import psycopg

from database.migrate import _postgres_url
from database.settings import settings


QUERIES = {
    "status_logs": """
        SELECT count(*)
        FROM raw_logs
        WHERE dataset = 'statuspage_incidents'
    """,
    "status_logs_with_incident_report": """
        SELECT count(*)
        FROM raw_logs rl
        WHERE rl.dataset = 'statuspage_incidents'
          AND EXISTS (
              SELECT 1
              FROM raw_news rn
              WHERE rn.source_type = 'statuspage_incident'
                AND rn.raw_payload->>'incident_id' = rl.raw_payload->>'incident_id'
          )
    """,
    "status_incidents": """
        SELECT count(*)
        FROM raw_news
        WHERE source_type = 'statuspage_incident'
    """,
    "gdelt_provider_time_candidates": """
        WITH candidates AS (
            SELECT rl.id, count(rn.id) AS candidate_count
            FROM raw_logs rl
            JOIN raw_news rn
              ON rn.source_type = 'gdeltv2_event'
             AND rn.published_at BETWEEN rl.event_time - interval '24 hours'
                                     AND rl.event_time + interval '24 hours'
             AND (
                  rn.url ILIKE '%' || lower(rl.source) || '%'
                  OR rn.title ILIKE '%' || rl.source || '%'
                  OR rn.body ILIKE '%' || rl.source || '%'
             )
            WHERE rl.dataset = 'statuspage_incidents'
              AND rl.event_time IS NOT NULL
            GROUP BY rl.id
        )
        SELECT count(*), coalesce(sum(candidate_count), 0)
        FROM candidates
    """,
    "hackernews_provider_time_candidates": """
        WITH candidates AS (
            SELECT rl.id, count(rn.id) AS candidate_count
            FROM raw_logs rl
            JOIN raw_news rn
              ON rn.source_type = 'hackernews_story'
             AND rn.published_at BETWEEN rl.event_time - interval '30 days'
                                     AND rl.event_time + interval '30 days'
             AND rn.raw_payload->>'provider' = rl.source
            WHERE rl.dataset = 'statuspage_incidents'
              AND rl.event_time IS NOT NULL
            GROUP BY rl.id
        )
        SELECT count(*), coalesce(sum(candidate_count), 0)
        FROM candidates
    """,
    "google_news_provider_time_candidates": """
        WITH candidates AS (
            SELECT rl.id, count(rn.id) AS candidate_count
            FROM raw_logs rl
            JOIN raw_news rn
              ON rn.source_type = 'google_news_story'
             AND rn.published_at BETWEEN rl.event_time - interval '30 days'
                                     AND rl.event_time + interval '30 days'
             AND rn.raw_payload->>'provider' = rl.source
            WHERE rl.dataset = 'statuspage_incidents'
              AND rl.event_time IS NOT NULL
            GROUP BY rl.id
        )
        SELECT count(*), coalesce(sum(candidate_count), 0)
        FROM candidates
    """,
    "github_release_pairs": """
        SELECT count(*)
        FROM raw_logs gl
        WHERE gl.dataset = 'gharchive_open_source'
          AND EXISTS (
              SELECT 1
              FROM raw_news rn
              WHERE rn.source_type = 'github_release'
                AND rn.source = gl.source
                AND rn.published_at BETWEEN gl.event_time - interval '1 hour'
                                        AND gl.event_time + interval '1 hour'
        )
    """,
    "osv_advisory_pairs": """
        SELECT count(*)
        FROM raw_logs ol
        WHERE ol.dataset = 'osv_advisories'
          AND EXISTS (
              SELECT 1
              FROM raw_news rn
              WHERE rn.source_type = 'osv_advisory'
                AND rn.raw_payload->>'advisory_id' = ol.raw_payload->>'advisory_id'
          )
    """,
}


def main() -> None:
    with psycopg.connect(_postgres_url(settings.database_url)) as conn:
        with conn.cursor() as cur:
            for name, query in QUERIES.items():
                cur.execute(query)
                print(f"{name}: {cur.fetchone()}")

            cur.execute(
                """
                SELECT source, count(*) AS logs
                FROM raw_logs
                WHERE dataset = 'statuspage_incidents'
                GROUP BY source
                ORDER BY logs DESC
                LIMIT 25
                """
            )
            print("top_statuspage_sources:")
            for source, logs in cur.fetchall():
                print(f"  {source}: {logs}")


if __name__ == "__main__":
    main()
