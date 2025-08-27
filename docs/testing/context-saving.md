# Playwright Context-Saving for AI-News (Next.js)

This guide documents fast, reliable Playwright testing for the AI-News Next.js app using saved browser context (storageState). The goal is to avoid logging in on every test, keep runs stable, and make it easy to support multiple roles if you add auth later.

## TL;DR
- Reuse Playwright `storageState` to skip repetitive logins.
- Keep a UI-driven login flow for full E2E checks; prefer env-driven selectors.
- Optionally support multiple roles with separate storage files and TTL.
- Never commit `.auth/` or any secrets.

## Layout
```
tests/
  e2e/
    auth.setup.ts          # Global setup to create storage state (saved context)
    test.config.ts         # Centralized config for selectors/paths
    lib/
      auth.ts              # Reusable login flow
      storage.ts           # Role-based storage helpers
    config/roles.ts        # Role credentials parsing (from env)
    fixtures.ts            # Authenticated + anonymous fixtures
    pages/                 # Page Objects
      LoginPage.ts
      DashboardPage.ts
    example.spec.ts        # Basic example
    role_example.spec.ts   # Role-based example (future-proof)
playwright.config.ts       # Playwright configuration
.auth/                     # Storage files (gitignored)
```

## Environment
- App URL: `UI_BASE_URL` (default `http://localhost:3000`).
- Login path: `LOGIN_PATH` (default `/login`).
- Selectors: `EMAIL_TESTID`, `PASSWORD_TESTID`, `SUBMIT_TESTID`.
- Roles (optional):
  - `PW_ROLES=admin,user`, `PW_DEFAULT_ROLE=user`
  - `TEST_EMAIL`/`TEST_PASSWORD` (user)
  - `TEST_ADMIN_EMAIL`/`TEST_ADMIN_PASSWORD` (admin)
- Storage TTL/refresh: `PW_STORAGE_TTL_HOURS=24`, `PW_REFRESH_STORAGE=1`.

## Strategies
- UI-driven: Use the login page once, save Playwright `storageState`, and reuse it.
- API-driven (faster): Call `/api/v1/auth/login` to get tokens and inject them via `addInitScript`, then save storage.

## Running
- Start dev server automatically via Playwright webServer or run `npm run dev`.
- API strategy: `PW_AUTH_STRATEGY=api TEST_EMAIL=... TEST_PASSWORD=... npx playwright test --config=tests/playwright.config.ts`.
- UI strategy: `PW_AUTH_STRATEGY=ui npx playwright test --config=tests/playwright.config.ts`.
- Headed: `npx playwright test --headed`.

## CI
See `.github/workflows/playwright-e2e.yml`.
- Lints and builds the Next.js app on every push/PR.
- E2E job runs only when `RUN_E2E` repository variable is `true` or via manual dispatch.
- Provide credentials via repo secrets for context creation.
- Prunes stale `.auth/` files older than `PW_STORAGE_TTL_HOURS` (default 24h).

## Env/Role-Scoped Storage + Sidecar Metadata
- Storage path: `tests/.auth/<env>/<role>-v<schema>.json` where `<env>` is derived from `UI_BASE_URL` host.
- Sidecar metadata: `<file>.meta.json` records env, role, schema, claims hash, createdAt, lastValidatedAt.
- Drift detection: optional call to `/api/v1/auth/me` computes claims hash to detect role/permission changes.

## Tips
- Use `data-testid` attributes for selectors.
- Keep tests small and independent; rely on fixtures to manage auth.
- When login flow or permissions change, delete `.auth/` or set `PW_REFRESH_STORAGE=1`.
