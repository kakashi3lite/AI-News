import prisma from './db.js';
import { parseJsonArray } from './utils.js';

/**
 * Watchlist engine — tracks companies/entities and matches incoming
 * articles to them via alias/keyword scoring. Match reasons are stored
 * so the dashboard can show *why* a story was attributed.
 */

// Default seed: AI/tech players. Edit or extend via /api/watchlist or the UI.
export const DEFAULT_WATCHLIST = [
  { name: 'Nvidia', aliases: ['NVDA', 'Jensen Huang'], keywords: ['gpu', 'cuda', 'blackwell', 'hopper', 'datacenter'], category: 'technology' },
  { name: 'OpenAI', aliases: ['Open AI', 'ChatGPT', 'GPT-5'], keywords: ['sam altman', 'gpt', 'sora', 'chatgpt'], category: 'technology' },
  { name: 'Microsoft', aliases: ['MSFT', 'MS'], keywords: ['azure', 'copilot', 'xbox', 'activision', 'windows', 'surface'], category: 'technology' },
  { name: 'Google', aliases: ['Alphabet', 'GOOGL', 'GOOG', 'DeepMind'], keywords: ['gemini', 'android', 'chrome', 'tpu', 'waymo', 'youtube', 'alphafold'], category: 'technology' },
  { name: 'Meta', aliases: ['Facebook', 'Instagram', 'WhatsApp', 'META'], keywords: ['llama', 'threads', 'reality labs', 'horizon os'], category: 'technology' },
  { name: 'Amazon', aliases: ['AMZN', 'AWS'], keywords: ['aws', 'bedrock', 'alexa', 'zoox', 'kindle', 'whole foods'], category: 'technology' },
  { name: 'Apple', aliases: ['AAPL'], keywords: ['iphone', 'ipad', 'macbook', 'vision pro', 'app store', 'siri', 'tim cook'], category: 'technology' },
  { name: 'Tesla', aliases: ['TSLA'], keywords: ['cybertruck', 'model y', 'full self driving', 'fsd', 'optimus', 'megapack', 'elon musk'], category: 'technology' },
];

/**
 * Seed the watchlist. Creates missing items and syncs aliases/keywords for the
 * default items (so default keyword updates propagate). User-added items that
 * are not in the defaults are left untouched.
 */
export async function seedWatchlist(items = DEFAULT_WATCHLIST) {
  const results = [];
  for (const item of items) {
    results.push(
      await prisma.watchlistItem.upsert({
        where: { name: item.name },
        update: {
          aliases: JSON.stringify(item.aliases || []),
          keywords: JSON.stringify(item.keywords || []),
          category: item.category || null,
        },
        create: {
          name: item.name,
          aliases: JSON.stringify(item.aliases || []),
          keywords: JSON.stringify(item.keywords || []),
          feeds: JSON.stringify(item.feeds || []),
          category: item.category || null,
        },
      })
    );
  }
  return results;
}

export async function getWatchlistItems() {
  return prisma.watchlistItem.findMany({ orderBy: { name: 'asc' } });
}

function escapeRegExp(str) {
  return String(str).replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

/**
 * Score an article against a watchlist item.
 * @returns {{ score: number, matchedOn: string[] }}
 */
export function matchArticleToItem(item, article) {
  const title = String(article.title || '').toLowerCase();
  const desc = String(article.description || '').toLowerCase();
  const aliases = parseJsonArray(item.aliases);
  const keywords = parseJsonArray(item.keywords);
  const matchedOn = [];
  let score = 0;

  const bump = (term, weight) => {
    score += weight;
    matchedOn.push(term);
  };

  for (const term of [item.name, ...aliases]) {
    const t = String(term).toLowerCase().trim();
    if (t.length < 2) continue;
    const re = new RegExp(`\\b${escapeRegExp(t)}\\b`);
    if (re.test(title)) bump(t, 3);
    else if (re.test(desc)) bump(t, 2);
  }

  for (const term of keywords) {
    const t = String(term).toLowerCase().trim();
    if (t.length < 2) continue;
    const re = new RegExp(`\\b${escapeRegExp(t)}\\b`);
    if (re.test(title)) bump(t, 2);
    else if (re.test(desc)) bump(t, 1.5);
  }

  return {
    score: Math.round(score * 10) / 10,
    matchedOn: [...new Set(matchedOn)].slice(0, 8),
  };
}

/** Link a persisted article to every matching watchlist item. Returns link count. */
export async function linkArticleToWatchlist(article) {
  const items = await getWatchlistItems();
  let count = 0;
  for (const item of items) {
    const { score, matchedOn } = matchArticleToItem(item, article);
    // Require a reasonably strong signal to avoid false positives
    // (e.g. "Prime Minister" matching Amazon's old 'prime' keyword).
    if (score >= 3) {
      await prisma.watchlistArticle.upsert({
        where: {
          watchlistItemId_articleId: { watchlistItemId: item.id, articleId: article.id },
        },
        update: { matchScore: score, matchedOn: JSON.stringify(matchedOn) },
        create: {
          watchlistItemId: item.id,
          articleId: article.id,
          matchScore: score,
          matchedOn: JSON.stringify(matchedOn),
        },
      });
      count += 1;
    }
  }
  return count;
}

/**
 * Recompute ALL watchlist links from scratch. Call after keywords change
 * or seed updates so stale/missed matches are corrected (accuracy first).
 */
export async function relinkAllWatchlist() {
  const items = await getWatchlistItems();
  const articles = await prisma.article.findMany({ select: { id: true, title: true, description: true } });
  await prisma.watchlistArticle.deleteMany({});

  let total = 0;
  for (const article of articles) {
    for (const item of items) {
      const { score, matchedOn } = matchArticleToItem(item, article);
      if (score >= 3) {
        await prisma.watchlistArticle.create({
          data: {
            watchlistItemId: item.id,
            articleId: article.id,
            matchScore: score,
            matchedOn: JSON.stringify(matchedOn),
          },
        });
        total += 1;
      }
    }
  }
  return { items: items.length, articles: articles.length, links: total };
}
