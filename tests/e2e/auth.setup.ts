import { chromium, type FullConfig } from '@playwright/test';
import fs from 'fs';
import path from 'path';
import { getTestConfig } from './test.config';
import { getRolesConfig } from './config/roles';
import { ensureStorageForRole, storagePathFor } from './lib/storage';

async function ensureDir(p: string) {
  await fs.promises.mkdir(path.dirname(p), { recursive: true });
}

async function globalSetup(_config: FullConfig) {
  const storagePath = path.resolve(__dirname, '../.auth/user.json');
  const cfg = getTestConfig();
  const { defaultRole, roles } = getRolesConfig();

  const roleNames = Object.keys(roles);
  if (roleNames.length === 0) {
    // No creds in env; create anonymous default storage for continuity
    await ensureDir(storagePath);
    const browser = await chromium.launch({ headless: true });
    const context = await browser.newContext();
    try { await context.storageState({ path: storagePath }); } finally { await browser.close(); }
    return;
  }

  for (const role of roleNames) {
    const creds = roles[role];
    await ensureStorageForRole(role, cfg, creds);
  }

  // Also keep legacy `.auth/user.json` pointing at default role for compatibility
  try {
    const defPath = storagePathFor(defaultRole);
    await ensureDir(storagePath);
    await fs.promises.copyFile(defPath, storagePath);
  } catch (e) {
    console.warn('[auth.setup] Could not write default storage:', e);
  }
}

export default globalSetup;

