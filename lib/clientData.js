/**
 * Unified client data layer.
 *
 * Two modes:
 *  - Server mode (local dev / Vercel): talks to the live API + DB.
 *  - Static mode (GitHub Pages): reads pre-built JSON snapshots from
 *    NEXT_PUBLIC_STATIC=true, with client-side filtering. The snapshots are
 *    generated at build time by scripts/generate-static-data.mjs.
 */

const STATIC = process.env.NEXT_PUBLIC_STATIC === 'true';
const BASE = process.env.NEXT_PUBLIC_BASE_PATH || '';

export const isStaticMode = STATIC;

async function fetchJson(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Failed to load ${url} (${res.status})`);
  return res.json();
}

export async function fetchNews({ query = '', category = '', tag = '', limit = 30 } = {}) {
  if (STATIC) {
    const data = await fetchJson(`${BASE}/data/news.json`);
    let list = data.articles || [];
    if (category) list = list.filter((a) => a.category === category);
    if (tag) list = list.filter((a) => (a.tags || []).includes(tag));
    if (query) {
      const q = query.toLowerCase();
      list = list.filter(
        (a) =>
          (a.title || '').toLowerCase().includes(q) ||
          (a.description || '').toLowerCase().includes(q)
      );
    }
    return { articles: list.slice(0, limit) };
  }

  const params = new URLSearchParams({ limit: String(limit) });
  if (query) params.set('q', query);
  if (category) params.set('category', category);
  if (tag) params.set('tag', tag);
  const res = await fetch(`/api/news?${params}`);
  return res.json();
}

export async function fetchThemes() {
  if (STATIC) return fetchJson(`${BASE}/data/themes.json`);
  const res = await fetch('/api/themes');
  return res.json();
}

export async function fetchWatchlist() {
  if (STATIC) return fetchJson(`${BASE}/data/watchlist.json`);
  const res = await fetch('/api/watchlist');
  return res.json();
}

export async function fetchDigest() {
  if (STATIC) return fetchJson(`${BASE}/data/digest.json`);
  const res = await fetch('/api/digest');
  return res.json();
}

export async function fetchCrossword() {
  if (STATIC) return fetchJson(`${BASE}/data/crossword.json`);
  const res = await fetch('/api/crossword');
  return res.json();
}

export async function fetchMeta() {
  if (STATIC) return fetchJson(`${BASE}/data/meta.json`);
  const res = await fetch('/api/health');
  return { generatedAt: new Date().toISOString() };
}

export async function triggerIngest() {
  if (STATIC) return { running: false, stats: null, static: true };
  const res = await fetch('/api/ingest', { method: 'POST' });
  return res.json();
}
