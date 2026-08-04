# Deploy to Cloudflare (Workers + D1) — server mode

The dashboard runs in two modes:

| Mode | Where | Data |
| --- | --- | --- |
| **Static** (default, live now) | GitHub Pages | JSON snapshots baked at build |
| **Server** (this guide) | Cloudflare Workers + D1 | Live API + real-time ingestion |

## Why Cloudflare D1 (and not Hostinger)

- **D1 is serverless SQLite** — our schema already is SQLite, so it ports almost 1:1.
- Free tier (5 GB, 5M reads/day) — enough for real-time competitive intel.
- Edge-global latency + **Cron Triggers** for scheduled ingestion — truly real-time.
- Zero server management (Hostinger VPS works but needs OS/DB/uptime ops).

## Prerequisites

```bash
npm install -g wrangler
npx wrangler login          # one-time auth to your Cloudflare account
```

## 1. Create the database

```bash
npx wrangler d1 create market-signal-db
# → copy the printed database_id into wrangler.toml
npx wrangler d1 execute market-signal-db --file=prisma/d1-migration.sql
npx wrangler d1 execute market-signal-db --file=prisma/d1-seed.sql
```

`prisma/d1-seed.sql` should `INSERT INTO WatchlistItem` rows for your
`DEFAULT_WATCHLIST` (generate with `npm run db:seed` logic or copy from
`lib/watchlist.js`).

## 2. Secrets

```bash
npx wrangler secret put CRON_SECRET        # e.g. openssl rand -hex 32
npx wrangler secret put OPENAI_API_KEY     # optional
npx wrangler secret put SMTP_HOST
npx wrangler secret put SMTP_USER
npx wrangler secret put SMTP_PASS
npx wrangler secret put SMTP_FROM
npx wrangler secret put DIGEST_EMAILS      # comma-separated
```

## 3. Build & deploy (Next.js on Workers)

```bash
npm i -D @cloudflare/next-on-pages @prisma/adapter-d1
# wire the D1 adapter in lib/db.js (see lib/db.d1.example.js)
npx @cloudflare/next-on-pages
npx wrangler pages deploy .vercel/output/static
# or: npx wrangler deploy   (with the OpenNext worker)
```

## 4. Verify

- `GET https://<your-subdomain>.workers.dev/api/health`
- `POST /api/ingest` → live ingestion from the edge
- Cron Triggers run `/api/cron/ingest` (30 min) and `/api/cron/digest` (07:00)

## Portability notes (already handled)

- `lib/utils.js` uses **no `node:crypto`** — hashing is a portable FNV-1a 64-bit
  implementation (works in Node, browsers, and Workers).
- All data-access logic is plain `fetch`-based; nothing is Node-only.

## Fallback: Hostinger VPS

If you prefer Hostinger: install Node 20 + run the server mode with the SQLite
DB (`DATABASE_URL="file:./dev.db"`), and add `systemd` timers for the two cron
commands. Works, but requires server upkeep.
