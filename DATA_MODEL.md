# Data Model

## News Entry Schema (`schemas/news-entry.schema.json`)
- `id` – SHA-256 hash identifier
- `title` – article title
- `url` – original article URL
- `source` – name, type, reliability, category
- `publishedAt` / `ingestedAt` – ISO timestamps
- `content.summary` – AI-generated summary
- `topics[]` – topic name with confidence
- `metadata.qualityScore` – float 0-1
- Optional PII: `authors` within `metadata.citations`

Retention policy: news entries recommended to be purged or archived after 30 days unless flagged `high-impact`.

## API Models
- `SummarizeRequest` → `SummarizeResponse`
- Batch variants for multiple articles
