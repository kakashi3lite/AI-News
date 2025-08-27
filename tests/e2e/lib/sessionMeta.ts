import fs from 'fs';
import path from 'path';

export type SessionMeta = {
  env: string;
  role: string;
  schema: number;
  claimsHash?: string;
  createdAt: string;
  lastValidatedAt: string;
};

export function envKey(baseURL: string): string {
  try {
    const u = new URL(baseURL);
    return u.host || baseURL;
  } catch {
    return baseURL;
  }
}

export function metaPathFor(storagePath: string): string {
  const dir = path.dirname(storagePath);
  const base = path.basename(storagePath, path.extname(storagePath));
  return path.join(dir, `${base}.meta.json`);
}

export function readMeta(storagePath: string): SessionMeta | undefined {
  const p = metaPathFor(storagePath);
  try { return JSON.parse(fs.readFileSync(p, 'utf8')); } catch { return undefined; }
}

export function writeMeta(storagePath: string, meta: SessionMeta) {
  const p = metaPathFor(storagePath);
  fs.mkdirSync(path.dirname(p), { recursive: true });
  fs.writeFileSync(p, JSON.stringify(meta, null, 2));
}

