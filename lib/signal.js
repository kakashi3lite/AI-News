import prisma from './db.js';
import { serializeArticle, parseJsonArray } from './utils.js';
import { computeVerificationMap, enrichWithImpact, outlookFor } from './impact.js';

/**
 * Signal computation for the dashboard:
 * - getTopStories: ranked by recency + source reliability + sentiment magnitude,
 *   then enriched with impact score, verification level, and outlook.
 * - getThemesWithStories: theme clusters with representative stories.
 * - getWatchlistWithStories: per-company news streams.
 * - getDigest: the daily "what matters today" summary.
 */

function computeSignalScore(a) {
  const ageHours = (Date.now() - new Date(a.publishedAt).getTime()) / 3600000;
  const recency = Math.exp(-ageHours / 24);
  const reliability = a.reliabilityScore ?? 0.7;
  const sentimentBoost = Math.abs(a.sentimentScore ?? 0) * 30;
  return Math.round((0.65 * reliability + 0.35 * recency) * 100 + sentimentBoost);
}

async function getWatchlistNames() {
  const items = await prisma.watchlistItem.findMany({ select: { name: true } });
  return items.map((i) => i.name);
}

export async function getTopStories(opts = {}) {
  const { limit = 20, sinceHours = 48, category, query, watchlistId, tag } = opts;
  const since = new Date(Date.now() - sinceHours * 3600000);
  const where = { publishedAt: { gte: since } };
  if (category) where.category = category;
  if (watchlistId) where.watchlistLinks = { some: { watchlistItemId: Number(watchlistId) } };

  const rows = await prisma.article.findMany({
    where,
    include: { source: true, summaries: { where: { summaryType: 'standard' } } },
    orderBy: { publishedAt: 'desc' },
    take: 300,
  });

  let list = rows.map((a) => ({ ...serializeArticle(a), signalScore: computeSignalScore(a) }));

  if (tag) {
    list = list.filter((a) => a.tags.some((t) => t.toLowerCase() === tag.toLowerCase()));
  }
  if (query) {
    const q = query.toLowerCase();
    list = list.filter(
      (a) => a.title.toLowerCase().includes(q) || a.description.toLowerCase().includes(q)
    );
  }

  list.sort((a, b) => b.signalScore - a.signalScore);
  const top = list.slice(0, limit);

  const [verificationMap, watchlistNames] = await Promise.all([
    computeVerificationMap(),
    getWatchlistNames(),
  ]);
  return enrichWithImpact(top, { verificationMap, watchlistNames });
}

export async function getThemesWithStories({ limit = 10, storiesPerTheme = 6 } = {}) {
  const themes = await prisma.theme.findMany({
    orderBy: [{ articleCount: 'desc' }, { velocity: 'desc' }],
    take: limit,
    include: { articles: true },
  });

  const [verificationMap, watchlistNames] = await Promise.all([
    computeVerificationMap(),
    getWatchlistNames(),
  ]);

  const results = [];
  for (const theme of themes) {
    const articleIds = theme.articles.map((t) => t.articleId);
    const stories = await prisma.article.findMany({
      where: { id: { in: articleIds } },
      include: { source: true, summaries: { where: { summaryType: 'standard' } } },
    });
    stories.sort((a, b) => new Date(b.publishedAt) - new Date(a.publishedAt));

    const serialized = stories.slice(0, storiesPerTheme).map(serializeArticle);
    const enriched = await enrichWithImpact(serialized, { verificationMap, watchlistNames });

    // Theme-level aggregate: average impact + outlook from top stories.
    const avgImpact = enriched.length
      ? Math.round(enriched.reduce((s, x) => s + (x.impactScore || 0), 0) / enriched.length)
      : 0;
    const topStory = enriched.find((e) => e.verification === 'verified') || enriched[0];
    const aggregateLabel = avgImpact >= 70 ? 'high' : avgImpact >= 45 ? 'medium' : 'low';

    results.push({
      id: String(theme.id),
      name: theme.name,
      slug: theme.slug,
      category: theme.category,
      articleCount: theme.articleCount,
      velocity: theme.velocity,
      sentimentScore: theme.sentimentScore,
      sentimentLabel: theme.sentimentLabel,
      impactScore: avgImpact,
      impactLabel: aggregateLabel,
      outlook: topStory
        ? outlookFor({
            verification: topStory.verification,
            impact: aggregateLabel,
            sentimentLabel: theme.sentimentLabel,
            category: theme.category,
            corroboration: topStory.corroboration || 1,
          })
        : 'Monitor.',
      stories: enriched,
    });
  }
  return results;
}

export async function getWatchlistWithStories({ storiesPerItem = 6 } = {}) {
  const items = await prisma.watchlistItem.findMany({
    orderBy: { name: 'asc' },
    include: { articleLinks: true, _count: { select: { articleLinks: true } } },
  });

  const [verificationMap, watchlistNames] = await Promise.all([
    computeVerificationMap(),
    getWatchlistNames(),
  ]);

  const results = [];
  for (const item of items) {
    const articleIds = item.articleLinks.map((l) => l.articleId);
    const articles = await prisma.article.findMany({
      where: { id: { in: articleIds } },
      include: { source: true, summaries: { where: { summaryType: 'standard' } } },
    });
    const linkMap = new Map(item.articleLinks.map((l) => [l.articleId, l]));
    articles.sort((a, b) => new Date(b.publishedAt) - new Date(a.publishedAt));

    const stories = articles.slice(0, storiesPerItem).map((a) => {
      const link = linkMap.get(a.id);
      return {
        ...serializeArticle(a),
        matchedOn: link ? parseJsonArray(link.matchedOn) : [],
        matchScore: link?.matchScore ?? 0,
      };
    });
    const enriched = await enrichWithImpact(stories, { verificationMap, watchlistNames });

    const avgImpact = enriched.length
      ? Math.round(enriched.reduce((s, x) => s + (x.impactScore || 0), 0) / enriched.length)
      : 0;
    const positive = enriched.filter((s) => s.sentimentLabel === 'positive').length;
    const negative = enriched.filter((s) => s.sentimentLabel === 'negative').length;

    results.push({
      id: String(item.id),
      name: item.name,
      aliases: parseJsonArray(item.aliases),
      keywords: parseJsonArray(item.keywords),
      category: item.category,
      articleCount: item._count.articleLinks,
      impactScore: avgImpact,
      sentimentTrend: positive > negative ? 'positive' : negative > positive ? 'negative' : 'neutral',
      stories: enriched,
    });
  }
  return results;
}

export async function getDigest() {
  const [themes, stories, watchlist] = await Promise.all([
    getThemesWithStories({ limit: 8, storiesPerTheme: 3 }),
    getTopStories({ limit: 10, sinceHours: 24 }),
    prisma.watchlistItem.findMany({
      orderBy: { name: 'asc' },
      include: { _count: { select: { articleLinks: true } } },
    }),
  ]);

  return {
    generatedAt: new Date().toISOString(),
    themes,
    stories,
    watchlist: watchlist.map((w) => ({
      id: String(w.id),
      name: w.name,
      category: w.category,
      articleCount: w._count.articleLinks,
    })),
  };
}
