import { test } from 'node:test';
import assert from 'node:assert/strict';

process.env.DATABASE_URL = process.env.DATABASE_URL || 'file:./dev.db';
const { computeImpact, verificationLabel, titleFingerprint, outlookFor } = await import('../../lib/impact.js');

test('verification: 3+ independent sources → verified', () => {
  assert.equal(verificationLabel(3, 0.7), 'verified');
  assert.equal(verificationLabel(4, 0.5), 'verified');
});

test('verification: 2 sources + high reliability → verified', () => {
  assert.equal(verificationLabel(2, 0.9), 'verified');
});

test('verification: 2 sources + lower reliability → developing', () => {
  assert.equal(verificationLabel(2, 0.6), 'developing');
});

test('verification: single source → unverified', () => {
  assert.equal(verificationLabel(1, 0.99), 'unverified');
});

test('impact: reliable, corroborated, market-moving story → high', () => {
  const r = computeImpact({
    reliability: 0.97,
    corroboration: 4,
    category: 'business',
    watchlistHit: true,
    ageHours: 1,
    sentimentScore: 0.8,
  });
  assert.equal(r.label, 'high');
  assert.ok(r.score >= 70);
});

test('impact: lone general story → low', () => {
  const r = computeImpact({
    reliability: 0.7,
    corroboration: 1,
    category: 'general',
    watchlistHit: false,
    ageHours: 48,
    sentimentScore: 0,
  });
  assert.equal(r.label, 'low');
});

test('impact: score is bounded to [0, 100]', () => {
  const r = computeImpact({ reliability: 2, corroboration: 99, category: 'business', watchlistHit: true, ageHours: -5, sentimentScore: 5 });
  assert.ok(r.score >= 0 && r.score <= 100);
});

test('outlook stays hedged for unverified stories', () => {
  const o = outlookFor({ verification: 'unverified', impact: 'high', sentimentLabel: 'negative', category: 'business', corroboration: 1 });
  assert.match(o.toLowerCase(), /monitor|caut|watch/);
});

test('titleFingerprint normalizes and truncates', () => {
  assert.equal(titleFingerprint('OpenAI Models Go Rogue in Cyber Tests!'), titleFingerprint('OpenAI models go rogue in cyber tests'));
  assert.notEqual(titleFingerprint('Apple earnings beat'), titleFingerprint('Tesla stock surges'));
});
