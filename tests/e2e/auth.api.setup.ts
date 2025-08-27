import path from 'path';
import fs from 'fs';
import { request, chromium, type FullConfig } from '@playwright/test';
import { getTestConfig } from './test.config';
import { getRolesConfig } from './config/roles';
import { storagePathFor } from './lib/storage';
import { envKey, writeMeta } from './lib/sessionMeta';

async function getTokensViaAPI(baseURL: string, email?: string, password?: string) {
  if (!email || !password) return undefined;
  const api = await request.newContext({ baseURL });
  const res = await api.post('/api/v1/auth/login', { data: { email, password } });
  if (!res.ok()) return undefined;
  const body = await res.json();
  return { access: body.access_token as string, refresh: body.refresh_token as string };
}

async function getClaimsHash(baseURL: string, accessToken: string | undefined) {
  if (!accessToken) return undefined;
  try {
    const api = await request.newContext({ baseURL, extraHTTPHeaders: { Authorization: `Bearer ${accessToken}` } });
    const res = await api.get('/api/v1/auth/me');
    if (!res.ok()) return undefined;
    const body = await res.json();
    const roles: string[] = (body.roles || []).map((r: string) => r.toLowerCase()).sort();
    const perms: string[] = (body.permissions || []).map((p: string) => p.toLowerCase()).sort();
    const joined = [...roles, ...perms].join('|');
    // Simple stable hash
    let hash = 0; for (let i=0;i<joined.length;i++){ hash = ((hash<<5)-hash)+joined.charCodeAt(i); hash|=0; }
    return String(hash);
  } catch { return undefined; }
}

export default async function globalSetup(_: FullConfig) {
  const cfg = getTestConfig();
  const rolesCfg = getRolesConfig();
  const env = envKey(cfg.baseURL);

  for (const role of Object.keys(rolesCfg.roles)) {
    const creds = rolesCfg.roles[role];
    const storagePath = path.resolve(__dirname, `../.auth/${env}/${role}-v1.json`);
    fs.mkdirSync(path.dirname(storagePath), { recursive: true });

    // Acquire tokens via API, or fallback to dummy tokens for local dev
    const tokens = await getTokensViaAPI(cfg.baseURL, creds.email, creds.password);
    const accessToken = tokens?.access;
    const refreshToken = tokens?.refresh || 'dev-refresh';

    const browser = await chromium.launch();
    const context = await browser.newContext();
    // Inject tokens for app to pick up (adjust keys to your app)
    context.addInitScript(({ accessToken, refreshToken }) => {
      localStorage.setItem('access_token', accessToken || 'dev-access');
      localStorage.setItem('refresh_token', refreshToken);
    }, { accessToken, refreshToken });

    const page = await context.newPage();
    await page.goto(cfg.baseURL);
    await context.storageState({ path: storagePath });

    // Sidecar metadata
    const claimsHash = await getClaimsHash(cfg.baseURL, accessToken || 'dev-access');
    writeMeta(storagePath, {
      env,
      role,
      schema: 1,
      claimsHash,
      createdAt: new Date().toISOString(),
      lastValidatedAt: new Date().toISOString(),
    });

    await browser.close();
  }

  // Legacy default pointer
  const defStorage = path.resolve(__dirname, '../.auth/user.json');
  const defRolePath = path.resolve(__dirname, `../.auth/${env}/${rolesCfg.defaultRole}-v1.json`);
  try { fs.copyFileSync(defRolePath, defStorage); } catch {}
}

