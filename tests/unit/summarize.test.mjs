import { test } from 'node:test';
import assert from 'node:assert/strict';

// summarize.js imports prisma — set the DB URL before dynamic import.
process.env.DATABASE_URL = process.env.DATABASE_URL || 'file:./dev.db';
const { extractiveSummary } = await import('../../lib/summarize.js');

test('extractiveSummary returns a non-empty deterministic summary', () => {
  const content =
    'The company reported record revenue in the latest quarter. Executives cited strong demand for AI chips. ' +
    'Analysts upgraded the stock after the earnings call. The company also announced a new data center product. ' +
    'Margins improved significantly across all business segments.';

  const a = extractiveSummary(content, 2);
  const b = extractiveSummary(content, 2);
  assert.equal(a, b, 'extractive summary must be deterministic');
  assert.ok(a.length > 0);
  assert.ok(a.length <= content.length);
});

test('short content is returned as-is (truncated to limit)', () => {
  const short = 'A brief update on market conditions today.';
  const out = extractiveSummary(short, 3);
  assert.ok(out.includes('market conditions'));
});

test('empty content returns empty string', () => {
  assert.equal(extractiveSummary(''), '');
  assert.equal(extractiveSummary(null), '');
});
