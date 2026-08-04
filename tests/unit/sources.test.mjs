import { test } from 'node:test';
import assert from 'node:assert/strict';
import { classifyArticle } from '../../lib/sources.js';

test('AI/tech headline classifies as technology', () => {
  assert.equal(
    classifyArticle('Nvidia Blackwell GPU demand surges as AI data centers expand', ''),
    'technology'
  );
});

test('finance headline classifies as business', () => {
  assert.equal(
    classifyArticle('Stock markets rally on strong earnings and revenue growth', ''),
    'business'
  );
});

test('unmatched headline falls back to general', () => {
  assert.equal(classifyArticle('A quiet day in the park for local communities', ''), 'general');
});
