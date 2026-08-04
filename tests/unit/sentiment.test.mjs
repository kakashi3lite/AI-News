import { test } from 'node:test';
import assert from 'node:assert/strict';
import { scoreSentiment } from '../../lib/sentiment.js';

test('positive text → positive label with score > 0', () => {
  const r = scoreSentiment('The company reported record growth and strong profits. Revenue surged and beat expectations.');
  assert.equal(r.label, 'positive');
  assert.ok(r.score > 0);
});

test('negative text → negative label with score < 0', () => {
  const r = scoreSentiment('The company faces a deepening crisis: losses, layoffs and a fraud scandal triggered a collapse.');
  assert.equal(r.label, 'negative');
  assert.ok(r.score < 0);
});

test('empty input → neutral', () => {
  const r = scoreSentiment('');
  assert.equal(r.label, 'neutral');
  assert.equal(r.score, 0);
});

test('balanced/mixed text stays neutral or near zero', () => {
  const r = scoreSentiment('The report covers market data, information and analysis of current conditions.');
  assert.equal(r.label, 'neutral');
});

test('score is bounded to [-1, 1]', () => {
  const r = scoreSentiment('growth growth growth growth growth growth growth growth');
  assert.ok(r.score <= 1);
});
