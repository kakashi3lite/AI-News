import { XMLParser } from 'fast-xml-parser';
import prisma from './db.js';
import { RSS_SOURCES, classifyArticle } from './sources.js';
import { articleHash, normalizeUrl, stripHtml, normalizeDate, truncate, wordCount } from './utils.js';
import { scoreSentiment } from './sentiment.js';
import { summarizeArticle } from './summarize.js';
import { seedWatchlist, linkArticleToWatchlist } from './watchlist.js';

/**
 * Market Signal ingestion pipeline (ESM, DB-backed).
 * 1. Fetch curated RSS feeds (real data, no API keys required)
 * 2. Normalize + classify + sentiment-score
 * 3. Dedup by stable urlHash and persist (SQLite)
 * 4. Link to watchlist items + generate summaries
 */

const xmlParser = new XMLParser({
  ignoreAttributes: false,
  attributeNamePrefix: '@_',
  processEntities: {
    enabled: true,
    maxTotalExpansions: 50000,
    maxEntitySize: 10000,
    maxExpandedLength: 200000,
    maxEntityCount: 50000,
  },
});
const MAX_PER_SOURCE = 25;
const MAX_CONCURRENT = 4;

let running = false;

export function isIngesting() {
  return running;
}

/** Stable title fingerprint for cross-source story dedup (Google News etc.). */
function titleFingerprint(title) {
  return String(title || '')
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, ' ')
    .split(/\s+/)
    .filter((w) => w.length > 3)
    .slice(0, 5)
    .join(' ');
}

function toArray(v) {
  return Array.isArray(v) ? v : v ? [v] : [];
}

function extractItems(parsed) {
  if (parsed.rss?.channel?.item) return toArray(parsed.rss.channel.item);
  if (parsed.feed?.entry) return toArray(parsed.feed.entry);
  return [];
}

function itemLink(item) {
  if (typeof item.link === 'string') return item.link;
  if (item.link?.['@_href']) return item.link['@_href'];
  if (item.id && typeof item.id === 'string') return item.id; // Atom id
  if (item.guid) return typeof item.guid === 'string' ? item.guid : item.guid['#text'];
  return '';
}

function itemImage(item) {
  if (item.enclosure?.['@_url']) return item.enclosure['@_url'];
  if (item['media:content']?.['@_url']) return item['media:content']['@_url'];
  if (item['media:thumbnail']?.['@_url']) return item['media:thumbnail']['@_url'];
  if (item.image?.url) return item.image.url;
  return '';
}

function itemPublished(item) {
  return item.pubDate || item.published || item.updated || item['dc:date'] || null;
}

function itemTags(item) {
  const tags = [];
  for (const c of toArray(item.category)) {
    const v = typeof c === 'string' ? c : c['#text'] || c._ || '';
    if (v) tags.push(String(v));
  }
  return tags;
}

function normalizeItem(item, src, sourceRow) {
  let title = stripHtml(item.title || '').slice(0, 500);
  const url = normalizeUrl(itemLink(item));
  const description = stripHtml(item.description || item.summary || item.content || '').slice(0, 3000);
  const content = stripHtml(
    item['content:encoded'] || item.content || item.summary || item.description || ''
  ).slice(0, 8000);

  // Google News titles carry attribution like "Headline - Source" or "Source: Headline".
  // Strip it so themes/watchlist match the actual headline, not the outlet.
  if (url.includes('news.google.com')) {
    title = title.replace(/\s*-\s*[A-Z][A-Za-z ]{2,40}$/, '').replace(/^[A-Z][A-Za-z ]{2,40}:\s*/, '').trim();
  }

  if (!title && !url) return null;

  const publishedAt = normalizeDate(itemPublished(item));
  const tags = itemTags(item).slice(0, 10);
  const category = classifyArticle(title, description);
  const sentiment = scoreSentiment(`${title}. ${description}`);
  const reliability = sourceRow?.reliabilityScore ?? 0.7;

  return {
    urlHash: articleHash({ url, title }),
    url,
    title,
    description,
    content,
    image: itemImage(item).slice(0, 1000),
    author:
      stripHtml(item['dc:creator'] || item.author?.['name'] || item.author || '').slice(0, 200) || null,
    publishedAt,
    category,
    tags,
    sentimentScore: sentiment.score,
    sentimentLabel: sentiment.label,
    reliabilityScore: reliability,
    wordCount: wordCount(`${title} ${description}`),
    sourceId: sourceRow?.id ?? null,
  };
}

async function ensureSources() {
  for (const src of RSS_SOURCES) {
    await prisma.source.upsert({
      where: { name: src.name },
      update: { url: src.url, category: src.category, reliabilityScore: src.reliability, isActive: true },
      create: {
        name: src.name,
        url: src.url,
        category: src.category,
        reliabilityScore: src.reliability,
        type: 'rss',
      },
    });
  }
  return prisma.source.findMany({ where: { isActive: true } });
}

async function fetchRss(url) {
  const res = await fetch(url, {
    headers: { 'User-Agent': 'MarketSignal/1.0 (market-intel-dashboard)' },
    signal: AbortSignal.timeout(12000),
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return xmlParser.parse(await res.text());
}

function chunk(array, size) {
  const out = [];
  for (let i = 0; i < array.length; i += size) out.push(array.slice(i, i + size));
  return out;
}

const delay = (ms) => new Promise((r) => setTimeout(r, ms));

/**
 * Run the full ingestion pipeline. Safe to call repeatedly (upsert + dedup).
 */
export async function ingestAll() {
  if (running) return { running: true, stats: null };
  running = true;
  const started = Date.now();
  const stats = {
    sourcesOk: 0,
    sourcesError: 0,
    fetched: 0,
    inserted: 0,
    duplicates: 0,
    linked: 0,
    summarized: 0,
    errors: [],
  };

  try {
    await seedWatchlist();
    const sources = await ensureSources();
    const sourceByName = new Map(sources.map((s) => [s.name, s]));

    const raw = [];
    for (const chunkOf of chunk(RSS_SOURCES, MAX_CONCURRENT)) {
      const results = await Promise.allSettled(chunkOf.map((src) => fetchRss(src.url)));
      results.forEach((res, i) => {
        const src = chunkOf[i];
        if (res.status === 'fulfilled') {
          stats.sourcesOk += 1;
          for (const item of extractItems(res.value).slice(0, MAX_PER_SOURCE)) {
            const art = normalizeItem(item, src, sourceByName.get(src.name));
            if (art) raw.push(art);
          }
        } else {
          stats.sourcesError += 1;
          stats.errors.push({ source: src.name, error: res.reason.message });
        }
      });
      await delay(250);
    }

    stats.fetched = raw.length;

    // Dedup against what's already persisted AND within this batch.
    // Same story arriving via multiple feeds (e.g. Google News syndication)
    // is collapsed using URL hash + a title fingerprint.
    const hashes = [...new Set(raw.map((a) => a.urlHash))];
    const existing = await prisma.article.findMany({
      where: { urlHash: { in: hashes } },
      select: { urlHash: true },
    });
    const seenHashes = new Set(existing.map((e) => e.urlHash));
    const seenTitles = new Set();

    const fresh = [];
    for (const a of raw) {
      const fp = titleFingerprint(a.title);
      const isGoogle = a.url.includes('news.google.com');
      const urlSeen = a.url ? seenHashes.has(a.urlHash) : false;
      const titleSeen = fp.length >= 5 && seenTitles.has(fp);
      // For Google News redirect links (unique per item), dedup by headline instead.
      if (urlSeen || (isGoogle && titleSeen)) continue;
      if (a.url) seenHashes.add(a.urlHash);
      if (fp.length >= 5) seenTitles.add(fp);
      fresh.push(a);
    }
    stats.duplicates = raw.length - fresh.length;

    if (fresh.length > 0) {
      await prisma.article.createMany({
        data: fresh.map((a) => ({
          urlHash: a.urlHash,
          url: a.url,
          title: a.title,
          description: a.description,
          content: a.content,
          imageUrl: a.image,
          author: a.author,
          publishedAt: a.publishedAt,
          category: a.category,
          tags: JSON.stringify(a.tags),
          sentimentScore: a.sentimentScore,
          sentimentLabel: a.sentimentLabel,
          reliabilityScore: a.reliabilityScore,
          wordCount: a.wordCount,
          sourceId: a.sourceId,
        })),
      });
      stats.inserted = fresh.length;

      const created = await prisma.article.findMany({
        where: { urlHash: { in: fresh.map((a) => a.urlHash) } },
      });

      // Watchlist linking.
      for (const article of created) {
        stats.linked += await linkArticleToWatchlist(article);
      }

      // Summaries: offline extractive for all; AI enhancement for top-reliability few.
      const topForAI = [...created]
        .sort((a, b) => (b.reliabilityScore || 0) - (a.reliabilityScore || 0))
        .slice(0, 5);
      for (const article of created) {
        const summary = await summarizeArticle(article, { withAI: topForAI.includes(article) });
        if (summary) stats.summarized += 1;
      }
    }

    await prisma.source.updateMany({
      where: { isActive: true },
      data: { lastFetchedAt: new Date() },
    });
  } catch (err) {
    console.error('[ingest] failed:', err);
    stats.errors.push({ source: 'main', error: err.message });
  } finally {
    running = false;
  }

  stats.processingTimeMs = Date.now() - started;
  return { running: false, stats };
}

/**
 * Ensure the DB has at least minArticles of real data; ingests if not.
 * Used so the dashboard shows real stories on first visit.
 */
export async function ensureData(minArticles = 15) {
  if (running) return { running: true, skipped: false, stats: null };
  const count = await prisma.article.count();
  if (count >= minArticles) return { running: false, skipped: true, stats: null };
  return { running: false, skipped: false, stats: (await ingestAll()).stats };
}
