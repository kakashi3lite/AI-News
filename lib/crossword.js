import prisma from './db.js';
import { fnv1a64 } from './utils.js';

/**
 * Daily Crossword — generated free, each day, from that day's real news.
 *
 * - Word list comes from today's watchlist companies, trending themes, and
 *   capitalized proper nouns in current headlines.
 * - A deterministic greedy placement builds the grid (stable per date).
 * - Clues are real sentences from the news with the answer blanked out,
 *   so solving the puzzle doubles as reading today's news.
 */

const STOPWORDS = new Set(
  'and the with from have been this that what when where who news says said after before about their there they were will would which while into your over under only just also even both more most some such then them than these those below above today year week'.split(' ')
);

function seededRandom(seedStr) {
  let a = parseInt(fnv1a64(seedStr).slice(0, 8), 16) || 1;
  return function () {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function shuffle(arr, rnd) {
  const a = [...arr];
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(rnd() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

function cleanWord(w) {
  const m = String(w || '').replace(/[^A-Za-z]/g, '');
  if (m.length < 3 || m.length > 12) return null;
  if (STOPWORDS.has(m.toLowerCase())) return null;
  return m;
}

async function collectWords(limit = 12) {
  const since = new Date(Date.now() - 48 * 3600000);
  const [articles, watchlist, themes] = await Promise.all([
    prisma.article.findMany({
      where: { publishedAt: { gte: since } },
      select: { title: true, description: true },
      orderBy: { publishedAt: 'desc' },
      take: 150,
    }),
    prisma.watchlistItem.findMany({ select: { name: true } }),
    prisma.theme.findMany({ select: { name: true }, orderBy: { articleCount: 'desc' }, take: 10 }),
  ]);

  const candidates = [];
  const seen = new Set();
  const add = (w) => {
    const c = cleanWord(w);
    if (c && !seen.has(c)) {
      seen.add(c);
      candidates.push(c);
    }
  };

  watchlist.forEach((w) => add(w.name));
  themes.forEach((t) => add(t.name));
  for (const a of articles) {
    const names = (a.title || '').match(/\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)?\b/g) || [];
    names.forEach((w) => add(w.replace(/\s/g, '')));
  }
  if (candidates.length < 6) {
    for (const a of articles) {
      for (const w of (a.title || '').match(/[A-Za-z]{4,12}/g) || []) add(w);
    }
  }

  return shuffle(candidates.slice(0, 30), seededRandom(String(limit))).slice(0, limit);
}

function sentenceWithWord(text, word) {
  const sentences = String(text || '').split(/(?<=[.!?])\s+/);
  const re = new RegExp(`\\b${word}\\b`, 'i');
  return sentences.find((s) => re.test(s)) || null;
}

function buildClue(word, articles, { isCompany, isTheme }) {
  for (const a of articles) {
    const src = sentenceWithWord(a.title, word) || sentenceWithWord(a.description, word);
    if (src) {
      const clue = src.replace(new RegExp(`\\b${word}\\b`, 'gi'), '_____');
      if (clue.trim().length > 10) return clue.trim();
    }
  }
  if (isCompany) return 'A company in today\u2019s market news.';
  if (isTheme) return 'A trending topic in today\u2019s news.';
  return 'A term from today\u2019s headlines.';
}

// ---------- grid placement (deterministic greedy) ----------

function canPlace(word, r, c, dir, grid, size) {
  const dr = dir === 'down' ? 1 : 0;
  const dc = dir === 'across' ? 1 : 0;
  if (r < 0 || c < 0) return false;
  if (r + dr * (word.length - 1) >= size || c + dc * (word.length - 1) >= size) return false;

  for (let i = 0; i < word.length; i++) {
    const cell = grid[r + dr * i][c + dc * i];
    if (cell && cell !== word[i]) return false;
  }
  for (let i = 0; i < word.length; i++) {
    const rr = r + dr * i;
    const cc = c + dc * i;
    const crossingHere = grid[rr][cc] === word[i];
    if (dir === 'across') {
      if (rr > 0 && grid[rr - 1][cc] && !crossingHere) return false;
      if (rr < size - 1 && grid[rr + 1][cc] && !crossingHere) return false;
    } else {
      if (cc > 0 && grid[rr][cc - 1] && !crossingHere) return false;
      if (cc < size - 1 && grid[rr][cc + 1] && !crossingHere) return false;
    }
  }
  // word boundaries (don't extend an existing word)
  const br = r - dr;
  const bc = c - dc;
  if (br >= 0 && bc >= 0 && grid[br][bc]) return false;
  const er = r + dr * word.length;
  const ec = c + dc * word.length;
  if (er < size && ec < size && grid[er][ec]) return false;

  return true;
}

function scorePlacement(word, r, c, dir, grid, size) {
  const dr = dir === 'down' ? 1 : 0;
  const dc = dir === 'across' ? 1 : 0;
  let intersections = 0;
  for (let i = 0; i < word.length; i++) {
    if (grid[r + dr * i][c + dc * i]) intersections += 1;
  }
  const center = (size - 1) / 2;
  const dist = Math.abs(r - center) + Math.abs(c - center);
  return intersections * 20 - dist;
}

function generateGrid(words, size = 12) {
  const grid = Array.from({ length: size }, () => Array(size).fill(null));
  const placed = [];
  const sorted = [...words].sort((a, b) => b.length - a.length);

  const tryPlace = (word, dir) => {
    let best = null;
    for (let r = 0; r < size; r++) {
      for (let c = 0; c < size; c++) {
        if (!canPlace(word, r, c, dir, grid, size)) continue;
        const s = scorePlacement(word, r, c, dir, grid, size);
        if (!best || s > best.score) best = { r, c, dir, score: s };
      }
    }
    return best;
  };

  // First word: across near the top-left.
  if (sorted[0]) {
    const w = sorted[0];
    const r = 1;
    const c = 1;
    for (let i = 0; i < w.length; i++) grid[r][c + i] = w[i];
    placed.push({ word: w, row: r, col: c, dir: 'across' });
  }

  for (const w of sorted.slice(1)) {
    const across = tryPlace(w, 'across');
    const down = tryPlace(w, 'down');
    const best = across && down
      ? (across.score >= down.score ? across : down)
      : across || down;
    if (!best) continue;
    const dr = best.dir === 'down' ? 1 : 0;
    const dc = best.dir === 'across' ? 1 : 0;
    for (let i = 0; i < w.length; i++) grid[best.r + dr * i][best.c + dc * i] = w[i];
    placed.push({ word: w, row: best.r, col: best.c, dir: best.dir });
  }

  return { grid, placed };
}

export function buildPuzzle(words, { size = 12 } = {}) {
  const upper = [...new Set(words.map((w) => String(w).toUpperCase()))].filter(
    (w) => w.length >= 3 && w.length <= 12
  );
  const { grid, placed } = generateGrid(upper, size);

  // Number word starts in reading order (row-major).
  const starts = placed
    .map((p, i) => ({ ...p, number: i + 1 }))
    .sort((a, b) => a.row - b.row || a.col - b.col)
    .map((p, i) => ({ ...p, number: i + 1 }));

  const cells = {};
  const numbers = {};
  starts.forEach((p) => {
    const dr = p.dir === 'down' ? 1 : 0;
    const dc = p.dir === 'across' ? 1 : 0;
    numbers[`${p.row},${p.col}`] = p.number;
    for (let i = 0; i < p.word.length; i++) {
      cells[`${p.row + dr * i},${p.col + dc * i}`] = p.word[i];
    }
  });

  return {
    size,
    cells,
    numbers,
    across: starts.filter((p) => p.dir === 'across'),
    down: starts.filter((p) => p.dir === 'down'),
    wordCount: starts.length,
  };
}

export async function getDailyCrossword({ date } = {}) {
  const day = date || new Date().toISOString().slice(0, 10);

  const since = new Date(Date.now() - 48 * 3600000);
  const [articles, watchlist, themes] = await Promise.all([
    prisma.article.findMany({
      where: { publishedAt: { gte: since } },
      select: { title: true, description: true },
      orderBy: { publishedAt: 'desc' },
      take: 150,
    }),
    prisma.watchlistItem.findMany({ select: { name: true } }),
    prisma.theme.findMany({ select: { name: true }, orderBy: { articleCount: 'desc' }, take: 10 }),
  ]);

  const companySet = new Set(watchlist.map((w) => String(w.name).toLowerCase()));
  const themeSet = new Set(themes.map((t) => String(t.name).toLowerCase()));

  const raw = await collectWords(14);
  const words = raw.map((w) => w.toUpperCase());

  const puzzle = buildPuzzle(words, { size: 12 });

  // Attach clues.
  const withClues = (list) =>
    list.map((p) => ({
      number: p.number,
      word: p.word,
      row: p.row,
      col: p.col,
      dir: p.dir,
      clue: buildClue(p.word, articles, {
        isCompany: companySet.has(p.word.toLowerCase()),
        isTheme: themeSet.has(p.word.toLowerCase()),
      }),
    }));

  return {
    date: day,
    size: puzzle.size,
    cells: puzzle.cells,
    numbers: puzzle.numbers,
    wordCount: puzzle.wordCount,
    across: withClues(puzzle.across),
    down: withClues(puzzle.down),
    generatedFrom: 'today\u2019s real news headlines',
  };
}
