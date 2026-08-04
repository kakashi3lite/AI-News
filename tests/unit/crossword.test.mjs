import { test } from 'node:test';
import assert from 'node:assert/strict';

process.env.DATABASE_URL = process.env.DATABASE_URL || 'file:./dev.db';
const { buildPuzzle } = await import('../../lib/crossword.js');

const WORDS = ['OPENAI', 'MICROSOFT', 'GOOGLE', 'WORKDAY', 'SAP', 'LATTICE', 'PERSONIO', 'SLACK'];

test('buildPuzzle places words into a valid grid', () => {
  const p = buildPuzzle(WORDS, { size: 14 });
  assert.equal(p.size, 14);
  assert.ok(p.wordCount >= 4, `expected at least 4 words placed, got ${p.wordCount}`);
  // every placed word must exist in cells with correct letters
  for (const w of [...p.across, ...p.down]) {
    for (let i = 0; i < w.word.length; i++) {
      const k = w.dir === 'across' ? `${w.row},${w.col + i}` : `${w.row + i},${w.col}`;
      assert.equal(p.cells[k], w.word[i], `cell ${k} should be ${w.word[i]}`);
    }
  }
  // every filled cell belongs to at least one word
  const allWordCells = new Set();
  for (const w of [...p.across, ...p.down]) {
    for (let i = 0; i < w.word.length; i++) {
      allWordCells.add(w.dir === 'across' ? `${w.row},${w.col + i}` : `${w.row + i},${w.col}`);
    }
  }
  for (const k of Object.keys(p.cells)) assert.ok(allWordCells.has(k), `orphan cell ${k}`);
});

test('buildPuzzle is deterministic for the same word list', () => {
  const a = buildPuzzle(WORDS, { size: 14 });
  const b = buildPuzzle(WORDS, { size: 14 });
  assert.deepEqual(a.cells, b.cells);
  assert.deepEqual(a.numbers, b.numbers);
});

test('words are uppercased and invalid ones are skipped', () => {
  const p = buildPuzzle(['OPENAI', 'x', 'TOOLONGWORD123456', 'OK', 'GOOGLE'], { size: 12 });
  for (const w of [...p.across, ...p.down]) {
    assert.match(w.word, /^[A-Z]{3,12}$/);
  }
});

test('crossing words share letters correctly (intersections are consistent)', () => {
  const p = buildPuzzle(WORDS, { size: 14 });
  // For every pair of words that intersect, the shared cell must hold the same letter
  const words = [...p.across, ...p.down];
  for (let i = 0; i < words.length; i++) {
    for (let j = i + 1; j < words.length; j++) {
      const wi = words[i];
      const wj = words[j];
      // collect cells of wi
      const cells = new Map();
      for (let x = 0; x < wi.word.length; x++) {
        cells.set(wi.dir === 'across' ? `${wi.row},${wi.col + x}` : `${wi.row + x},${wi.col}`, wi.word[x]);
      }
      for (let x = 0; x < wj.word.length; x++) {
        const k = wj.dir === 'across' ? `${wj.row},${wj.col + x}` : `${wj.row + x},${wj.col}`;
        if (cells.has(k)) {
          assert.equal(cells.get(k), wj.word[x], `intersection mismatch at ${k}`);
        }
      }
    }
  }
});
