import prisma from './db.js';
import { parseJsonArray, slugify } from './utils.js';
import { RSS_SOURCES } from './sources.js';

// Source names (and their significant words) must never become themes —
// e.g. "Guardian" or "Times" appearing in a headline is attribution, not a story.
const SOURCE_TERMS = new Set(
  RSS_SOURCES.flatMap((s) =>
    String(s.name)
      .toLowerCase()
      .split(/[^a-z]+/)
      .filter((w) => w.length >= 4)
  )
);

/**
 * DB-backed theme/trend extraction for the Market Signal dashboard.
 * Clusters recent articles by significant terms, then stores/refreshes
 * Theme + ThemeArticle rows with aggregate sentiment and velocity.
 */

const STOPWORDS = new Set(
  'a an and are as at be but by for from has have in into is it its of on or that the this to was were will with about after against all also amid among around before being between both business can come could day days during each even every first from get has had have her his how however into its just last latest like made make many may more most much must new news next not now off one only other our over own per people press said says she should since some still such than that the their them then there these they this those through time today under until upon very was way week weeks well were what when where which while who whom will with within without would year years you your company companies report reports quarter quarterly million billion percent market markets object objects added adding including share shares stock stocks price prices month months result results latest three four five six seven eight nine ten plus two first second third world state states'.split(' ')
);

const MIN_ARTICLES = 2;
const MAX_THEMES = 15;

function buildTermIndex(articles) {
  const map = new Map(); // term -> Set(articleId)
  for (const a of articles) {
    const rawTitle = String(a.title || '');
    const title = rawTitle.toLowerCase();
    const desc = String(a.description || '').toLowerCase();
    const text = `${title} ${desc}`;
    const terms = new Set();

    for (const tag of parseJsonArray(a.tags)) {
      const t = String(tag).toLowerCase().trim();
      // Skip source-attribution tags (e.g. Google News emits "NBC News", "Australia News").
      if (/news$/i.test(t)) continue;
      if (SOURCE_TERMS.has(t)) continue;
      if (t.length >= 3 && !STOPWORDS.has(t)) terms.add(t);
    }

    // Significant title words (min 5 chars, not stopwords, not source names).
    const words = title
      .replace(/[^a-z0-9\s]/g, ' ')
      .split(/\s+/)
      .filter((w) => w.length >= 5 && !STOPWORDS.has(w) && !SOURCE_TERMS.has(w));
    words.forEach((w) => terms.add(w));

    // Uppercase acronyms (e.g. "AI", "GDP", "IPO") from the raw title.
    const acro = (rawTitle.match(/\b[A-Z]{2,}\b/g) || []).map((w) => w.toLowerCase());
    acro.forEach((w) => terms.add(w));

    for (const term of terms) {
      if (term.length >= 3 && text.includes(term)) {
        if (!map.has(term)) map.set(term, new Set());
        map.get(term).add(a.id);
      }
    }
  }
  return map;
}

function dominantCategory(articles) {
  const counts = {};
  for (const a of articles) {
    const c = a.category || 'general';
    counts[c] = (counts[c] || 0) + 1;
  }
  return Object.entries(counts).sort((x, y) => y[1] - x[1])[0]?.[0] || 'general';
}

/**
 * Extract themes from articles ingested in the last `windowHours`.
 * Recomputes Theme rows + links; returns the refreshed themes.
 */
export async function extractThemesFromDb({ windowHours = 24 } = {}) {
  const now = Date.now();
  const since = new Date(now - windowHours * 3600000);
  const prevSince = new Date(now - windowHours * 2 * 3600000);

  const [current, previous] = await Promise.all([
    prisma.article.findMany({
      where: { publishedAt: { gte: since } },
      select: { id: true, title: true, description: true, tags: true, sentimentScore: true, sentimentLabel: true, category: true },
    }),
    prisma.article.findMany({
      where: { publishedAt: { gte: prevSince, lt: since } },
      select: { id: true, title: true, description: true, tags: true },
    }),
  ]);

  if (current.length < MIN_ARTICLES) return [];

  const currentByTerm = buildTermIndex(current);
  const previousByTerm = buildTermIndex(previous);

  const candidates = [];
  for (const [term, articleIds] of currentByTerm) {
    if (articleIds.size < MIN_ARTICLES) continue;
    // Terms that cover more than half the window are background noise, not themes.
    if (articleIds.size > current.length * 0.5) continue;
    const prevCount = previousByTerm.get(term)?.size || 0;
    candidates.push({ term, count: articleIds.size, velocity: articleIds.size - prevCount, articleIds });
  }

  candidates.sort((a, b) => b.count - a.count || b.velocity - a.velocity);
  const top = candidates.slice(0, MAX_THEMES);

  const results = [];
  for (const t of top) {
    const articles = current.filter((a) => t.articleIds.has(a.id));
    const scores = articles.map((a) => a.sentimentScore ?? 0).filter((n) => !Number.isNaN(n));
    const avgScore = scores.length ? scores.reduce((s, n) => s + n, 0) / scores.length : 0;
    let label = 'neutral';
    if (avgScore > 0.1) label = 'positive';
    else if (avgScore < -0.1) label = 'negative';

    const theme = await prisma.theme.upsert({
      where: { slug: slugify(t.term) },
      update: {
        name: t.term,
        articleCount: t.count,
        velocity: t.velocity,
        sentimentScore: Math.round(avgScore * 100) / 100,
        sentimentLabel: label,
        keywords: JSON.stringify([t.term]),
        category: dominantCategory(articles),
      },
      create: {
        name: t.term,
        slug: slugify(t.term),
        articleCount: t.count,
        velocity: t.velocity,
        sentimentScore: Math.round(avgScore * 100) / 100,
        sentimentLabel: label,
        keywords: JSON.stringify([t.term]),
        category: dominantCategory(articles),
      },
    });

    // Refresh theme↔article links.
    await prisma.themeArticle.deleteMany({ where: { themeId: theme.id } });
    if (articles.length > 0) {
      await prisma.themeArticle.createMany({
        data: articles.map((a) => ({ themeId: theme.id, articleId: a.id, score: 1 })),
      });
    }

    results.push({
      id: String(theme.id),
      name: theme.name,
      slug: theme.slug,
      category: theme.category,
      articleCount: theme.articleCount,
      velocity: theme.velocity,
      sentimentScore: theme.sentimentScore,
      sentimentLabel: theme.sentimentLabel,
    });
  }

  // Remove themes that are no longer detected so the dashboard stays accurate
  // to the current window (stale clusters like old noise terms disappear).
  const detectedSlugs = results.map((r) => r.slug);
  if (detectedSlugs.length > 0) {
    await prisma.theme.deleteMany({ where: { slug: { notIn: detectedSlugs } } });
  }

  return results;
}
