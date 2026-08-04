import { test } from 'node:test';
import assert from 'node:assert/strict';
import { articleHash, stripHtml, slugify, normalizeUrl, serializeArticle } from '../../lib/utils.js';

test('articleHash is stable for the same URL', () => {
  const a = articleHash({ url: 'https://example.com/story', title: 'x' });
  const b = articleHash({ url: 'https://example.com/story', title: 'x' });
  assert.equal(a, b);
});

test('articleHash differs for different URLs', () => {
  assert.notEqual(
    articleHash({ url: 'https://example.com/a', title: 'x' }),
    articleHash({ url: 'https://example.com/b', title: 'x' })
  );
});

test('stripHtml decodes numeric entities like &#8217;', () => {
  assert.equal(stripHtml('Nvidia doesn&#8217;t mess around'), 'Nvidia doesn\u2019t mess around');
});

test('stripHtml removes tags and normalizes whitespace', () => {
  assert.equal(stripHtml('<p>Hello  <b>world</b></p>'), 'Hello world');
});

test('slugify produces safe slugs', () => {
  assert.equal(slugify('Earnings Season 2026!'), 'earnings-season-2026');
});

test('normalizeUrl strips query and hash', () => {
  assert.equal(normalizeUrl('https://a.com/x?utm=1#frag'), 'https://a.com/x');
});

test('serializeArticle parses tags JSON and picks standard summary', () => {
  const row = {
    id: 1,
    title: 'T',
    description: 'D',
    url: 'https://a.com',
    imageUrl: null,
    publishedAt: new Date('2026-01-01'),
    fetchedAt: new Date('2026-01-01'),
    category: 'business',
    tags: '["ai","finance"]',
    sentimentScore: 0.5,
    sentimentLabel: 'positive',
    reliabilityScore: 0.9,
    source: { name: 'Bloomberg', url: 'u', reliabilityScore: 0.96 },
    summaries: [{ summaryType: 'standard', summaryText: 'S', modelUsed: 'extractive', confidence: 0.55 }],
  };
  const s = serializeArticle(row);
  assert.deepEqual(s.tags, ['ai', 'finance']);
  assert.equal(s.summary, 'S');
  assert.equal(s.summaryModel, 'extractive');
  assert.equal(s.source.name, 'Bloomberg');
});
