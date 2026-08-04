# Market Signal — Next Steps

## Working Prototype (done ✅)
- [x] Real RSS ingestion → SQLite (Prisma), no API keys required
- [x] Dedup, sentiment scoring, source reliability
- [x] Watchlist + theme extraction + daily digest
- [x] Dashboard UI: Signal / Watchlist / Stories / Digest / Tools / **Crossword** / For You
- [x] Impact + verification engine (corroboration-based, market-impact score, outlooks)
- [x] Auto-mode: learned interest weights + auto-follow chips + impact-aware recommendations
- [x] Daily news crossword (free, generated from real headlines, 10 words/day)
- [x] Local login (email+PIN vault) — bookmarks, personal watchlist, history, For You
- [x] Worksy competitive watchlist seed (HR-tech/employee-experience market)
- [x] Email digest (nodemailer + /api/cron/digest, SMTP env-driven)
- [x] Vercel cron config (vercel.json) + CRON_SECRET-guarded endpoints
- [x] Cloudflare D1 package: wrangler.toml, d1-migration.sql, DEPLOY_CLOUDFLARE.md
- [x] Portable hashing (FNV-1a, no node:crypto) — ready for Workers/edge
- [x] 35 unit tests, zero ESLint warnings, server + static builds green
- [x] Deployed live: https://kakashi3lite.github.io/AI-News/

## Next (recommended order)
1. **Deploy server mode to Cloudflare** — follow `docs/DEPLOY_CLOUDFLARE.md` (D1 + Workers + cron) for true real-time ingestion + email digest from the edge.
2. **Email digest credentials** — add `SMTP_*` + `DIGEST_EMAILS` env vars to activate delivery (code + cron are ready).
3. **Historical charts** — per-company / per-theme sentiment + impact over time (data is persisted; needs a chart component).
4. **Alerting** — watchlist/impact thresholds → webhook or email trigger.
5. **PWA** — offline + installable (nice for the crossword habit).

## Verification commands
```bash
npm run dev                              # run the dashboard
npm run test:unit                        # 35 unit tests
node --env-file=.env scripts/test-ingest.mjs   # full pipeline smoke test
node --env-file=.env scripts/maintenance.mjs   # retire/relink/themes/crossword
npm run build                            # server-mode production build
bash scripts/build-static.sh /AI-News    # static export for GitHub Pages
```

## API surface (real data)
```
GET  /api/news          # top stories (impact + verification enriched)
GET  /api/themes        # theme clusters (+impact, +stories)
GET  /api/watchlist     # tracked companies + matched stories (POST to add)
GET  /api/digest        # today's digest
GET  /api/crossword     # daily news crossword (date param)
POST /api/ingest        # manual refresh (rate-limited)
GET  /api/cron/ingest   # scheduled ingestion (Bearer: CRON_SECRET)
GET  /api/cron/digest   # scheduled digest email (Bearer: CRON_SECRET)
```

**Next Review:** Schedule regular project reviews
## Deployed (done ✅)
- [x] GitHub Pages live: https://kakashi3lite.github.io/AI-News/
- [x] main synced to kakashi3lite/AI-News (old lineage on legacy/ai-news branch)
- [x] Login + personalized "For You" verified on the live site

## To refresh live data
```bash
node --env-file=.env scripts/generate-static-data.mjs   # re-ingest + snapshot
bash scripts/build-static.sh /AI-News                    # rebuild out/
cd out && git add -A && git commit -am "refresh data" && git push -f origin gh-pages
```
