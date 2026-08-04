import prisma from './db.js';

/**
 * Impact & verification engine.
 *
 * Verification: a story is corroborated when multiple INDEPENDENT sources cover
 * the same headline (matched by title fingerprint). Combined with the source's
 * reliability score this yields: verified / developing / unverified.
 *
 * Impact: a transparent 0–100 score estimating a story's effect on global
 * markets = reliability + corroboration + market relevance + watchlist weight
 * + recency + sentiment magnitude.
 *
 * Outlook: a hedged, rule-based "what happens next" note per story.
 */

/** Stable title fingerprint used to group the same story across sources. */
export function titleFingerprint(title) {
  return String(title || '')
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, ' ')
    .split(/\s+/)
    .filter((w) => w.length > 3)
    .slice(0, 5)
    .join(' ');
}

/**
 * Map articleId → corroboration count (distinct sources covering same story)
 * over the recent window.
 */
export async function computeVerificationMap({ windowHours = 48 } = {}) {
  const since = new Date(Date.now() - windowHours * 3600000);
  const articles = await prisma.article.findMany({
    where: { publishedAt: { gte: since } },
    select: { id: true, title: true, source: { select: { name: true } } },
  });

  const groups = new Map();
  for (const a of articles) {
    const fp = titleFingerprint(a.title);
    if (!fp) continue;
    if (!groups.has(fp)) groups.set(fp, { sources: new Set(), ids: [] });
    groups.get(fp).sources.add(a.source?.name || 'unknown');
    groups.get(fp).ids.push(a.id);
  }

  const map = new Map();
  for (const [, g] of groups) {
    const corroboration = g.sources.size;
    for (const id of g.ids) map.set(id, corroboration);
  }
  return map;
}

export function verificationLabel(corroboration, reliability) {
  if (corroboration >= 3) return 'verified';
  if (corroboration === 2 && (reliability || 0) >= 0.85) return 'verified';
  if (corroboration >= 2) return 'developing';
  return 'unverified';
}

export function computeImpact({
  reliability = 0.7,
  corroboration = 1,
  category = 'general',
  watchlistHit = false,
  ageHours = 48,
  sentimentScore = 0,
}) {
  const reliabilityWeight = (reliability || 0.7) * 40;
  const corroborationWeight = Math.min(corroboration, 5) * 6;
  const marketWeight = category === 'business' || category === 'technology' ? 15 : 5;
  const watchlistWeight = watchlistHit ? 10 : 0;
  const recency = Math.max(0, (24 - ageHours) / 24) * 5;
  const sentimentMagnitude = Math.min(Math.abs(sentimentScore || 0), 1) * 5;

  const score = Math.min(
    100,
    Math.max(0, Math.round(reliabilityWeight + corroborationWeight + marketWeight + watchlistWeight + recency + sentimentMagnitude))
  );
  let label = 'low';
  if (score >= 70) label = 'high';
  else if (score >= 45) label = 'medium';
  return { score, label };
}

export function outlookFor({ verification, impact, sentimentLabel, category, corroboration }) {
  if (!verification || verification === 'unverified' || corroboration <= 1) {
    return 'Single-source report — monitor for corroboration.';
  }
  if (impact === 'high' && sentimentLabel === 'negative') {
    return 'High-impact negative — likely headwinds for related markets near-term.';
  }
  if (impact === 'high' && sentimentLabel === 'positive') {
    return 'High-impact positive — likely tailwind for related sectors near-term.';
  }
  if (sentimentLabel === 'positive') return 'Positive signal — modest tailwind expected.';
  if (sentimentLabel === 'negative') return 'Negative signal — modest headwind expected.';
  return 'Neutral signal — monitor for follow-through.';
}

/**
 * Enrich a list of serialized articles with verification, corroboration,
 * impact score/label, and an outlook note.
 */
export async function enrichWithImpact(articles, { verificationMap, watchlistNames = [] } = {}) {
  const map = verificationMap || (await computeVerificationMap());
  const names = watchlistNames.map((n) => String(n).toLowerCase());

  return articles.map((a) => {
    const corroboration = map.get(Number(a.id)) || 1;
    const verification = verificationLabel(corroboration, a.reliabilityScore);
    const text = `${a.title || ''} ${a.description || ''}`.toLowerCase();
    const watchlistHit = names.some((n) => n && text.includes(n));
    const ageHours = a.publishedAt
      ? (Date.now() - new Date(a.publishedAt).getTime()) / 3600000
      : 48;

    const impact = computeImpact({
      reliability: a.reliabilityScore,
      corroboration,
      category: a.category,
      watchlistHit,
      ageHours,
      sentimentScore: a.sentimentScore,
    });

    return {
      ...a,
      corroboration,
      verification,
      impactScore: impact.score,
      impactLabel: impact.label,
      outlook: outlookFor({
        verification,
        impact: impact.label,
        sentimentLabel: a.sentimentLabel,
        category: a.category,
        corroboration,
      }),
    };
  });
}
