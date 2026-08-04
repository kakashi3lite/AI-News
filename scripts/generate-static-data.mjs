// Generate static JSON snapshots for GitHub Pages / static hosting.
// This runs the REAL ingestion pipeline, then writes the results to
// public/data/*.json which the static site reads at runtime.
//
// Run:  node --env-file=.env scripts/generate-static-data.mjs
// Then: NEXT_PUBLIC_STATIC=true NEXT_PUBLIC_BASE_PATH=/AI-News npm run build
import fs from 'node:fs';
import path from 'node:path';
import prisma from '../lib/db.js';
import { ingestAll } from '../lib/ingest.js';
import { extractThemesFromDb } from '../lib/themes.js';
import { getDailyCrossword } from '../lib/crossword.js';
import {
  getTopStories,
  getThemesWithStories,
  getWatchlistWithStories,
  getDigest,
} from '../lib/signal.js';

const outDir = path.join(process.cwd(), 'public', 'data');
fs.mkdirSync(outDir, { recursive: true });

console.log('🌐 Running real ingestion (RSS → dedup → watchlist → summaries)…');
const ingest = await ingestAll();
console.log(
  `   inserted=${ingest.stats?.inserted} duplicates=${ingest.stats?.duplicates} sourcesOk=${ingest.stats?.sourcesOk}/${(ingest.stats?.sourcesOk ?? 0) + (ingest.stats?.sourcesError ?? 0)}`
);

console.log('🏷️  Extracting themes…');
await extractThemesFromDb({ windowHours: 24 });

const [stories, themes, watchlist, digest] = await Promise.all([
  getTopStories({ limit: 150, sinceHours: 72 }),
  getThemesWithStories({ limit: 12, storiesPerTheme: 5 }),
  getWatchlistWithStories({ storiesPerItem: 6 }),
  getDigest(),
]);

console.log('🧩 Building today\u2019s news crossword…');
const crossword = await getDailyCrossword({ date: new Date().toISOString().slice(0, 10) });

const meta = {
  generatedAt: new Date().toISOString(),
  sources: await prisma.source.count(),
  articles: await prisma.article.count(),
  note: 'Static snapshot for GitHub Pages. Rebuild + redeploy to refresh the data.',
};

fs.writeFileSync(path.join(outDir, 'news.json'), JSON.stringify({ articles: stories, generatedAt: meta.generatedAt }));
fs.writeFileSync(path.join(outDir, 'themes.json'), JSON.stringify({ themes, generatedAt: meta.generatedAt }));
fs.writeFileSync(path.join(outDir, 'watchlist.json'), JSON.stringify({ items: watchlist, generatedAt: meta.generatedAt }));
fs.writeFileSync(path.join(outDir, 'digest.json'), JSON.stringify({ ...digest, generatedAt: meta.generatedAt }));
fs.writeFileSync(path.join(outDir, 'crossword.json'), JSON.stringify(crossword));
fs.writeFileSync(path.join(outDir, 'meta.json'), JSON.stringify(meta, null, 2));

console.log(`✅ Static data written to ${outDir}`);
console.log(
  `   stories=${stories.length} themes=${themes.length} watchlist=${watchlist.length} crossword=${crossword.wordCount} words articles=${meta.articles}`
);

await prisma.$disconnect();
