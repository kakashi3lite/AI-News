import { test } from 'node:test';
import assert from 'node:assert/strict';

// watchlist.js imports prisma — set the DB URL before dynamic import.
process.env.DATABASE_URL = process.env.DATABASE_URL || 'file:./dev.db';
const { matchArticleToItem } = await import('../../lib/watchlist.js');

const NVIDIA = {
  id: 1,
  name: 'Nvidia',
  aliases: JSON.stringify(['NVDA', 'Jensen Huang']),
  keywords: JSON.stringify(['gpu', 'cuda', 'blackwell', 'hopper', 'datacenter']),
};

const AMAZON = {
  id: 2,
  name: 'Amazon',
  aliases: JSON.stringify(['AMZN', 'AWS']),
  keywords: JSON.stringify(['aws', 'bedrock', 'alexa', 'zoox', 'kindle', 'whole foods']),
};

const APPLE = {
  id: 3,
  name: 'Apple',
  aliases: JSON.stringify(['AAPL']),
  keywords: JSON.stringify(['iphone', 'ipad', 'macbook', 'vision pro', 'app store', 'siri', 'tim cook']),
};

test('company name in title is a strong match', () => {
  const r = matchArticleToItem(NVIDIA, { title: 'Nvidia shares jump after Blackwell demand surge', description: '' });
  assert.ok(r.score >= 3, `expected strong score, got ${r.score}`);
  assert.ok(r.matchedOn.includes('nvidia'));
});

test('keyword match alone (2 hits) can cross the threshold', () => {
  const r = matchArticleToItem(APPLE, { title: 'New iPhone and iPad launch dates leaked', description: '' });
  assert.ok(r.score >= 3, `expected >= 3, got ${r.score}`);
});

test('generic headline does NOT false-positive on Amazon', () => {
  const r = matchArticleToItem(AMAZON, { title: 'Prime Minister announces new economic policy today', description: '' });
  assert.ok(r.score < 3, `expected weak match, got ${r.score}`);
});

test('search-themed headline does NOT false-positive on Google', () => {
  const GOOGLE = {
    id: 4,
    name: 'Google',
    aliases: JSON.stringify(['Alphabet', 'GOOGL', 'DeepMind']),
    keywords: JSON.stringify(['gemini', 'android', 'chrome', 'tpu', 'waymo', 'youtube', 'alphafold']),
  };
  const r = matchArticleToItem(GOOGLE, { title: 'Major search under way for boy missing at sea', description: '' });
  assert.ok(r.score < 3, `expected weak match, got ${r.score}`);
});
