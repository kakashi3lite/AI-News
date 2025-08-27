import fs from 'fs';
import path from 'path';
import { chromium } from '@playwright/test';
import type { E2EConfig } from '../test.config';
import type { RoleCredentials } from '../config/roles';
import { performLogin } from './auth';

const AUTH_DIR = path.resolve(__dirname, '../../.auth');

export function storagePathFor(role: string): string {
  return path.join(AUTH_DIR, `role-${role}.json`);
}

function ttlExpired(file: string, hours: number): boolean {
  try {
    const stat = fs.statSync(file);
    const ageMs = Date.now() - stat.mtimeMs;
    return ageMs > hours * 3600 * 1000;
  } catch {
    return true;
  }
}

export async function ensureStorageForRole(role: string, cfg: E2EConfig, creds?: RoleCredentials) {
  const file = storagePathFor(role);
  const refresh = process.env.PW_REFRESH_STORAGE === '1';
  const ttlHours = Number(process.env.PW_STORAGE_TTL_HOURS || '24');

  if (!refresh && fs.existsSync(file) && !ttlExpired(file, ttlHours)) {
    return file;
  }

  await fs.promises.mkdir(path.dirname(file), { recursive: true });

  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext();
  const page = await context.newPage();
  try {
    if (creds && creds.email && creds.password) {
      await performLogin(page, cfg, { email: creds.email, password: creds.password });
    }
    await context.storageState({ path: file });
  } finally {
    await browser.close();
  }

  return file;
}

