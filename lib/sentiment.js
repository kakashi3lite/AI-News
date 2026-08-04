/**
 * Deterministic, offline lexicon-based sentiment scorer.
 * Always available (no API key), so the dashboard is accurate even offline.
 * Score ranges -1 (negative) … +1 (positive); neutral near 0.
 */

const POSITIVE = new Set([
  'breakthrough', 'surge', 'soar', 'record', 'growth', 'grow', 'profit', 'profitability',
  'gain', 'gains', 'rally', 'upbeat', 'positive', 'success', 'successful', 'win', 'wins',
  'boost', 'boosted', 'jump', 'jumped', 'rises', 'rise', 'rising', 'climb', 'climbs',
  'strong', 'stronger', 'beat', 'beats', 'outperform', 'opportunity', 'opportunities',
  'innovative', 'milestone', 'award', 'approval', 'approved', 'expansion', 'expand',
  'partnership', 'deal', 'investment', 'investing', 'momentum', 'accelerate', 'accelerating',
  'progress', 'improve', 'improves', 'improvement', 'recovery', 'rebound', 'bullish',
  'exceed', 'exceeds', 'upgrade', 'upgraded', 'launch', 'launches', 'hits', 'high',
  'best', 'top', 'leader', 'leadership', 'optimistic', 'confident', 'booming',
]);

const NEGATIVE = new Set([
  'crisis', 'collapse', 'plunge', 'plummet', 'drop', 'drops', 'fall', 'falls', 'falling',
  'decline', 'declines', 'loss', 'losses', 'layoff', 'layoffs', 'fired', 'lawsuit',
  'fraud', 'scandal', 'ban', 'banned', 'breach', 'hack', 'hacked', 'hacker', 'cyberattack',
  'slump', 'weak', 'weaker', 'slowdown', 'recession', 'inflation', 'tariff', 'tariffs',
  'downgrade', 'downgraded', 'warn', 'warns', 'warning', 'risk', 'risks', 'fear', 'fears',
  'panic', 'crash', 'crashed', 'bankrupt', 'bankruptcy', 'investigation', 'probe',
  'sanctions', 'war', 'conflict', 'death', 'deaths', 'fatal', 'lawsuit', 'sues', 'sued',
  'miss', 'misses', 'disappoint', 'disappointing', 'cut', 'cuts', 'slashes', 'halts',
  'halt', 'suspend', 'suspension', 'recall', 'recalls', 'fine', 'fined', 'penalty',
  'trouble', 'struggle', 'struggles', 'struggling', 'volatile', 'uncertain', 'uncertainty',
]);

const NEGATORS = new Set(['not', 'no', 'never', "don't", "doesn't", "isn't", 'without', 'despite', 'but']);

export function scoreSentiment(text) {
  if (!text) return { score: 0, label: 'neutral' };
  const words = String(text).toLowerCase().match(/[a-z']+/g) || [];
  let pos = 0;
  let neg = 0;
  let windowNegation = false;

  for (const w of words) {
    if (NEGATORS.has(w)) {
      windowNegation = true;
      continue;
    }
    if (POSITIVE.has(w)) pos += windowNegation ? -0.5 : 1;
    if (NEGATIVE.has(w)) neg += windowNegation ? -0.5 : 1;
    // reset negation window after a few words
    windowNegation = false;
  }

  const total = pos + neg;
  const score = total === 0 ? 0 : (pos - neg) / Math.max(1, pos + neg);
  let label = 'neutral';
  if (score > 0.15) label = 'positive';
  else if (score < -0.15) label = 'negative';

  return { score: round(score, 3), label };
}

function round(n, decimals) {
  const f = 10 ** decimals;
  return Math.round(n * f) / f;
}
