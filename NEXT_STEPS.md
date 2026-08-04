# Market Signal — Next Steps

## Working Prototype (done ✅)
- [x] Real RSS ingestion → SQLite (Prisma), no API keys required
- [x] Dedup, sentiment scoring, source reliability
- [x] Watchlist + theme extraction + daily digest
- [x] Dashboard UI: Signal / Watchlist / Stories / Digest / Tools
- [x] Build, lint, and 22 unit tests green

## Next (recommended order)
1. **Scheduled ingestion for production** — wire Vercel Cron to `GET /api/cron/ingest` with `CRON_SECRET` (or `node --env-file=.env scripts/test-ingest.mjs` in a local cron).
2. **Custom watchlist seed** — replace the AI/tech defaults in `lib/watchlist.js` with your real competitors, or add them via the Watchlist UI.
3. **Email digest delivery** — SMTP env vars are already defined; add a mailer step to the digest endpoint.
4. **Historical charts** — per-company / per-theme sentiment over time (data is already persisted; needs a chart component).
5. **Alerting** — watchlist keyword hits above a threshold → webhook/email trigger.
6. **Postgres migration** — the Prisma schema is portable; switch provider + `DATABASE_URL`.
7. **Deployment config** — pick a host (Vercel recommended); `netlify.toml` is currently mismatched with the build output.

## Verification commands
```bash
npm run dev                              # run the dashboard
node --env-file=.env scripts/test-ingest.mjs   # full pipeline smoke test
node --env-file=.env scripts/maintenance.mjs   # re-decode/relink/refresh themes
npm run test:unit                        # unit tests (Node test runner)
npm run build                            # production build
```

## API surface (real data)
```
GET  /api/news          # top stories (search/category/tag/watchlist filters)
GET  /api/themes        # trending theme clusters (+stories)
GET  /api/watchlist     # tracked companies + matched stories (POST to add)
GET  /api/digest        # today's digest
POST /api/ingest        # manual refresh (ingest + relink + themes)
GET  /api/cron/ingest   # scheduled ingestion (Bearer: CRON_SECRET)
```

**Next Review:** Schedule regular project reviews