# Market Signal — Competitive Intelligence Dashboard

A working competitive-intelligence web app: real-time market news, source-graded and
sentiment-tagged, with company watchlists, a personalized "For You" feed, and a daily
digest. Built on Next.js 14, SQLite (Prisma), and curated RSS feeds — **no API keys
required** for the core experience.

## Why it's accurate

- **Real data, always** — 13 curated RSS feeds (WSJ, Bloomberg, FT, BBC, NPR, Guardian, CNN, The Verge, TechCrunch, Hacker News, Google News) ingested with no keys.
- **Dedup by design** — stable URL hashes + title fingerprints collapse the same story across feeds and runs.
- **Source reliability** — every story carries its outlet's reliability score.
- **Offline sentiment** — deterministic lexicon scoring (positive/negative/neutral) on every story.
- **Match attribution** — watchlist stories show *why* they matched (the alias/keyword that fired).
- **Graceful AI** — extractive summaries always work; OpenAI enhancement is opt-in and never breaks the pipeline.

## Sign in & personalization

Create a local profile (email + PIN) to keep your **research on this device**:
- 🔖 **Bookmark stories** — build a saved-research library
- 📌 **Personal watchlist** — track any company, not just the defaults
- ⭐ **"For You" feed** — recommendations ranked from your saved stories, watchlist, and reading history
- 🕒 **Recently read** — your research trail

The profile is a local vault (device-scoped), so it works on fully static hosting.
PINs are hashed with WebCrypto (SHA-256 + salt); nothing is stored in plaintext.

## Quick Start (full app with live API + DB)

```bash
npm install
npx prisma migrate dev --name init   # creates SQLite DB (prisma/dev.db)
npm run dev                          # → http://localhost:3000/dashboard
```

First visit auto-ingests real RSS data. Use **Refresh data** to re-ingest
(fetch → dedup → watchlist link → summarize → themes).

### Optional AI summaries
Set `OPENAI_API_KEY` in `.env.local` for abstractive summaries on top stories
(falls back to extractive on failure/rate-limit).

## Dashboard

| View | What it does |
| --- | --- |
| **For You** | Personalized feed from your saved stories, watchlist, and history (after login) |
| **Signal** | Trending themes + top stories ranked by recency, reliability, sentiment |
| **Watchlist** | Tracked companies with matched stories + match reasons |
| **Stories** | Search, category tabs, tag chips over the full feed |
| **Digest** | One-scroll daily summary: theme pulse, watchlist pulse, top stories |
| **Tools** | YouTube summarizer (earnings calls / product launches) |

## Deploy to GitHub Pages (static)

The app runs in two modes: **server mode** (API + DB, above) and **static mode**
(pre-built JSON snapshots read by the client — no server needed).

```bash
# 1. Generate real data snapshots (runs the full ingestion pipeline)
node --env-file=.env scripts/generate-static-data.mjs

# 2. Build the static export (basePath = repo sub-path, e.g. /AI-News)
bash scripts/build-static.sh /AI-News

# 3. Serve locally to verify (simulates the /AI-News subpath)
node scripts/serve-static.mjs /AI-News 8080   # → http://localhost:8080/AI-News/dashboard
```

Then push `out/` to the `gh-pages` branch and enable Pages:

```bash
cd out && touch .nojekyll && git init -b gh-pages
git add -A && git commit -m "deploy: market signal static site"
git remote add origin https://github.com/<you>/<repo>.git
git push -f origin gh-pages
```

Refresh the data by re-running steps 1–2 and re-pushing `gh-pages`.

## API (server mode)

```
GET  /api/news          # top stories (q, category, tag, limit)
GET  /api/themes        # theme clusters + representative stories
GET  /api/watchlist     # companies + matched stories (POST to add)
GET  /api/digest        # daily digest
POST /api/ingest        # manual refresh (rate-limited)
GET  /api/cron/ingest   # scheduled ingestion (Bearer: CRON_SECRET)
```

## Architecture

```
app/MarketDashboard.js        # dashboard shell (views, login, refresh)
components/market/*           # views + cards + badges (For You, Signal, Watchlist…)
contexts/UserContext.jsx      # local user session + research vault
lib/client/userStore.js       # localStorage user vault (bookmarks/watchlist/history)
lib/clientData.js             # unified data layer (server API ↔ static JSON)
lib/ingest.js                 # RSS fetch → normalize → dedup → persist
lib/summarize.js              # extractive (offline) + optional OpenAI
lib/sentiment.js              # offline lexicon sentiment
lib/watchlist.js              # alias/keyword matching + seeding
lib/themes.js                 # DB-backed theme clusters + velocity
lib/signal.js                 # ranking, watchlist streams, digest
prisma/schema.prisma          # SQLite schema
scripts/generate-static-data.mjs  # build-time data snapshots
scripts/build-static.sh           # static export builder
scripts/serve-static.mjs          # local Pages-like server
tests/unit/*.test.mjs             # accuracy unit tests (Node test runner)
```

## Testing & Verification

```bash
npm run test:unit                  # 22 unit tests (sentiment, dedup, watchlist, themes)
node --env-file=.env scripts/test-ingest.mjs   # end-to-end pipeline smoke test
npm run lint                       # ESLint (zero warnings/errors)
npm run build                      # server-mode production build
```

## Security

- Ingest endpoint is rate-limited (20s window); scheduled ingestion requires `CRON_SECRET`.
- Watchlist POST validated with zod (length + type limits).
- All text sanitized/truncated at ingest; user input rendered via React (XSS-safe).
- Security headers set in server mode; no secrets shipped to the client in static mode.
- `.env*` and the SQLite DB are gitignored.

## Watchlist

Defaults are AI/tech players (Nvidia, OpenAI, Microsoft, Google, Meta, Amazon, Apple, Tesla).
Edit `lib/watchlist.js` → `DEFAULT_WATCHLIST` to track your real competitors, or add them
from the Watchlist UI (saved to your profile when signed in).

## Out of Scope (cut for the prototype)

Social features, A/B experiments, mock jobs/monitoring APIs, dead overlay components, and the
Python MLOps/DeployX suite. See `PROJECT_STATUS.md` and `NEXT_STEPS.md` for status and roadmap.

## License

MIT
