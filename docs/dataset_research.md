# Dataset Research: Logs to Relevant Tech News/Incidents

## Goal

Build a dataset for a system where a user enters an operational log or incident
update and the system returns relevant technology news, incident reports, or
project activity records.

The key requirement is not only volume. We need at least one defensible
relevance signal:

- same incident id;
- same provider/project plus close time window;
- same release/security advisory/project event;
- manually verified or semi-verified hard negatives.

## Current Local Dataset

Current database state:

- `raw_news`: 344 449 rows
- `raw_logs`: 9 210 rows
- total selected objects: 353 701 rows

Current `raw_news` sources:

- `gdeltv2_event`: 335 844
- `hackernews_story`: 3 147
- `google_news_story`: 2 212
- `osv_advisory`: 860
- `statuspage_incident`: 1 977
- `github_release`: 409

Current `raw_logs` sources:

- `statuspage_incidents`: 7 882
- `osv_advisories`: 860
- `gharchive_open_source`: 468

Relevance checks:

- 7 882 status update logs have direct linked incident reports.
- 5 805 status update logs have Hacker News provider/time candidates.
- 3 938 status update logs have Google News provider/time candidates.
- GDELT provider/time matching is available as a broad candidate-pool check,
  but it is intentionally not the primary relevance metric.
- 417 GitHub activity logs have matching GitHub release rows.
- 860 OSV package events have direct linked OSV advisories.

## Source Classes

### 1. Public Statuspage APIs

Statuspage incident APIs are the best core source for relevance.

Pattern:

```text
https://<status-domain>/api/v2/incidents.json
```

Data shape:

- incident title;
- impact/status;
- timestamps;
- affected components;
- incident updates;
- stable incident id.

Relevance strategy:

```text
raw_logs = incident updates
raw_news = incident report/summary
positive pair = same incident_id
```

Validated providers include:

- GitHub
- Cloudflare
- OpenAI
- Discord
- Reddit
- Datadog
- Atlassian
- Twilio
- SendGrid
- DigitalOcean
- Vercel
- Netlify
- Supabase
- Anthropic
- Shopify
- Zoom
- Dropbox
- Box
- Sentry
- CircleCI
- HashiCorp
- MongoDB
- Confluent
- Elastic
- npm
- PyPI
- Grafana
- NewRelic
- Snowflake
- Figma
- Canva
- Zapier
- HubSpot
- DropboxSign
- Miro
- Airtable
- LaunchDarkly
- 1Password
- Bitbucket
- Jira
- Confluence

Assessment:

- Relevance: high.
- Volume: medium.
- Best use: validation positives and demo.

### 2. Hacker News Algolia

Hacker News is useful for technology news and community discussion around
outages. Queries are targeted by provider:

```text
{provider} outage
{provider} incident
{provider} status
```

Example API:

```text
https://hn.algolia.com/api/v1/search_by_date?query=Cloudflare%20outage&tags=story
```

Relevance strategy:

```text
statuspage log
-> same provider
-> Hacker News story within a time window
```

Assessment:

- Relevance: medium.
- Volume: medium.
- Best use: external tech-news candidates and hard negatives.

### 3. GDELT 2.1 Events

GDELT is a large open news/event corpus. It updates frequently and has raw files.

Useful for:

- broad external news/event context;
- hard negatives;
- large candidate pools.

Less useful for:

- guaranteed relevance to SaaS incidents;
- specific provider matching for small outages.

Assessment:

- Relevance: low to medium without query filtering.
- Volume: very high.
- Best use: candidate pool, hard negatives, benchmark.

### 4. GH Archive

GH Archive records the public GitHub event timeline in hourly JSON archives.
It contains commits, issues, pull requests, releases, comments, forks, watches,
and other public activity.

Relevance strategy:

```text
raw_logs = GitHub activity events
raw_news = GitHub release rows
positive pair = same repository and close timestamp
```

Current result:

- 300 projects selected.
- 281 projects have inserted release/activity pairs.
- 409 release rows.
- 468 GitHub activity log rows.
- 417 release-pair matches.

Assessment:

- Relevance: medium for OSS project activity.
- Volume: very high if querying more hours/days.
- Best use: open-source project layer, not outage news.

### 5. OSV / GitHub Security Advisories / NVD

Security advisories are promising for open-source project relevance:

```text
package vulnerability report
-> advisory/news item
-> affected package/project
-> release/fix/reference
```

Candidate sources:

- OSV.dev API
- GitHub Security Advisory database
- NVD CVE data feeds

Assessment:

- Relevance: high for security incidents.
- Volume: high.
- Best next research target.
- Caveat: not availability logs; this is vulnerability/security relevance.

### 6. Loghub / Large Technical Log Datasets

Loghub-style datasets are useful for scale and benchmark, but weak for news
relevance.

Current local use:

- `bolu61/loghub_2`: 310 000 rows.

Assessment:

- Relevance: low.
- Volume: high.
- Best use: BM25/full-text/dense/HNSW latency benchmark only.

## Recommended Dataset Design

Use separate evaluation layers instead of mixing all rows as equally relevant.

### Core Relevance Set

Use Statuspage:

```text
statuspage update log -> same incident report
```

This is the cleanest positive-label source.

### Tech News Candidate Set

Use Hacker News:

```text
statuspage update log -> same provider + time-window HN story
```

These are candidates, not guaranteed positives.

### OSS Project Set

Use GH Archive:

```text
GitHub activity log -> same repo release
```

This supports the "open-source projects" angle.

### Broad News/Negative Set

Use GDELT:

```text
same time window, different provider/topic
```

This is useful for hard negatives and retrieval stress.

### Performance Benchmark Set

Use Loghub only for load:

```text
BM25 / full-text / dense latency / index size
```

Do not use Loghub for relevance evaluation.

## Next Expansion Plan

1. Add OSV.dev advisories.
   - Store advisories as `raw_news` with source type `osv_advisory`.
   - Store affected package/project records as `raw_logs`-like events or a new
     structured table.
   - Match by package ecosystem/name and disclosure/fix time.

2. Expand GH Archive to multiple days.
   - Query more hourly archives around known release dates.
   - Prefer projects with releases, issues, and HN/GDELT mentions.

3. Add curated postmortem/blog sources.
   - Cloudflare blog incidents.
   - Discord engineering postmortems.
   - GitHub/Atlassian/Vercel/Netlify incident writeups.
   - These are lower volume but high-quality explanations.

4. Build explicit validation pairs.
   - Positives: same incident id, same repo release, same advisory package.
   - Hard negatives: same time window but different provider/project.
   - Easy negatives: random other provider/project.

## Source Notes

- GH Archive records public GitHub timeline events in hourly archives and is
  available as raw JSON files and BigQuery public data.
- GDELT describes its data as free/open and exposes raw data files and BigQuery
  datasets; GDELT 2.0 updates every 15 minutes.
- Hacker News Algolia search API exposes searchable HN stories by date.
- OSV.dev is a public vulnerability database/API for open-source packages.
