# Market Signal — Project Status

## Current State
- **Status**: Working prototype (MVP) — functional end-to-end
- **Last Updated**: 2026-08-05
- **Focus**: Competitive / market intelligence dashboard with accurate, real data

## What Works (Verified)
- Real RSS ingestion (13 curated feeds, no API keys required) → SQLite via Prisma
- Dedup across runs and within batch (URL hash + title fingerprint)
- Sentiment scoring (offline lexicon) + source reliability scores on every story
- Extractive summaries (offline) with optional OpenAI enhancement (graceful fallback)
- Company watchlist with alias/keyword matching + match attribution
- Theme/trend extraction with velocity + aggregate sentiment
- Dashboard views: Today's Signal, Watchlist, Stories (search/filter), Daily Digest, Tools (YouTube)
- Cron endpoint `/api/cron/ingest` for scheduled ingestion
- `npm run build` ✓ · ESLint ✓ · 22 unit tests ✓

## Accuracy Guarantees
- Every story shows: source, reliability %, sentiment label, publish time
- Dedup verified (re-ingest adds 0 duplicates)
- Watchlist false-positive regression tests
- AI failures never break the pipeline (fall back to extractive)

## Cut (documented out of scope)
- Social features, A/B experiments, jobs/monitoring mock APIs, dead overlays
- Python MLOps/DeployX suite (unrelated to app runtime)
- Auth (single-user demo pattern)

## Verified Metrics
- Sources: 13/13 parsing, 0 errors on clean runs
- Articles: ~270 stored; re-ingest inserts only new stories
- Watchlist: 8 seeded companies with accurate attribution
- Build: exit 0 · Lint: exit 0 · Tests: 22/22 pass

## Known Limitations (prototype)
- Theme extraction can surface borderline noise terms
- Watchlist defaults re-sync on ingest (UI edits to default items are overwritten)
- OpenAI summarization may hit rate limits (falls back to extractive)
