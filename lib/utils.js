/**
 * Shared helpers for the Market Signal dashboard.
 * Kept dependency-free and deterministic so accuracy is reproducible.
 * IMPORTANT: no Node-only APIs here — this file is portable to
 * Cloudflare Workers / edge runtimes (browsers + Node + Workers).
 */

/** Portable FNV-1a 64-bit hex hash (deterministic, no node:crypto). */
export function fnv1a64(input) {
  const mask = 0xffffffffffffffffn;
  let hash = 0xcbf29ce484222325n;
  for (let i = 0; i < input.length; i++) {
    hash ^= BigInt(input.charCodeAt(i));
    hash = BigInt.asUintN(64, hash * 0x100000001b3n);
  }
  return hash.toString(16).padStart(16, '0');
}

/** Normalize a URL: drop fragment + query, strip trailing slashes. */
export function normalizeUrl(url) {
  if (!url) return '';
  try {
    const u = new URL(url);
    u.hash = '';
    u.search = '';
    return u.toString().replace(/\/+$/, '');
  } catch {
    return String(url).trim();
  }
}

/** Stable identity key for an article (URL when present, else title). */
export function articleKey({ url, title }) {
  const base = url ? normalizeUrl(url) : '';
  if (base) return `url:${base}`;
  return `title:${String(title || '').toLowerCase().trim()}`;
}

/** Hex digest used as the unique DB column (portable FNV-1a 64). */
export function articleHash({ url, title }) {
  return fnv1a64(articleKey({ url, title }));
}

export function stripHtml(text) {
  if (!text) return '';
  return String(text)
    .replace(/<[^>]*>/g, ' ')
    .replace(/&amp;/g, '&')
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>')
    .replace(/&quot;/g, '"')
    .replace(/&#39;|&apos;/g, "'")
    .replace(/&nbsp;/g, ' ')
    .replace(/&#(\d+);/g, (_, code) => String.fromCharCode(Number(code)))
    .replace(/&#x([0-9a-fA-F]+);/g, (_, code) => String.fromCharCode(parseInt(code, 16)))
    .replace(/\s+/g, ' ')
    .trim();
}

export function parseJsonArray(str, fallback = []) {
  try {
    const v = JSON.parse(str);
    return Array.isArray(v) ? v : fallback;
  } catch {
    return fallback;
  }
}

export function slugify(text) {
  return String(text)
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
    .slice(0, 80);
}

export function truncate(text, max = 300) {
  if (!text) return '';
  const s = String(text).trim();
  return s.length > max ? `${s.slice(0, max).trimEnd()}…` : s;
}

export function normalizeDate(input) {
  if (!input) return new Date();
  const d = new Date(input);
  return Number.isNaN(d.getTime()) ? new Date() : d;
}

export function timeAgo(date) {
  const diff = Date.now() - new Date(date).getTime();
  const mins = Math.floor(diff / 60000);
  if (mins < 1) return 'just now';
  if (mins < 60) return `${mins}m ago`;
  const hours = Math.floor(mins / 60);
  if (hours < 24) return `${hours}h ago`;
  return `${Math.floor(hours / 24)}d ago`;
}

export function clamp(n, min, max) {
  return Math.min(max, Math.max(min, n));
}

export function wordCount(text) {
  if (!text) return 0;
  return String(text).trim().split(/\s+/).filter(Boolean).length;
}

/** Serialize a Prisma article row into a clean client-facing object. */
export function serializeArticle(a) {
  const tags = parseJsonArray(a.tags);
  const standard = a.summaries?.find((s) => s.summaryType === 'standard');
  return {
    id: String(a.id),
    title: a.title,
    description: a.description || '',
    url: a.url,
    image: a.imageUrl || '',
    publishedAt: a.publishedAt ? new Date(a.publishedAt).toISOString() : null,
    fetchedAt: a.fetchedAt ? new Date(a.fetchedAt).toISOString() : null,
    category: a.category || 'general',
    tags,
    sentimentScore: a.sentimentScore ?? null,
    sentimentLabel: a.sentimentLabel || null,
    reliabilityScore: a.reliabilityScore ?? 0.7,
    source: a.source
      ? { name: a.source.name, url: a.source.url, reliabilityScore: a.source.reliabilityScore }
      : null,
    summary: standard ? standard.summaryText : null,
    summaryModel: standard ? standard.modelUsed : null,
    summaryConfidence: standard ? standard.confidence : null,
  };
}
