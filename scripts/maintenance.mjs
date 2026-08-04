// One-off maintenance: re-decode entities in stored articles, re-seed watchlist
// keywords, relink, and refresh themes.
// Run: node --env-file=.env scripts/maintenance.mjs
import prisma from '../lib/db.js';
import { stripHtml } from '../lib/utils.js';
import { seedWatchlist, relinkAllWatchlist, RETIRED_DEFAULTS } from '../lib/watchlist.js';
import { extractThemesFromDb } from '../lib/themes.js';
import { getDailyCrossword } from '../lib/crossword.js';

await seedWatchlist();

// Remove retired default watchlist items (old AI-tech defaults no longer tracked).
if (RETIRED_DEFAULTS.length > 0) {
  const removed = await prisma.watchlistItem.deleteMany({
    where: { name: { in: RETIRED_DEFAULTS } },
  });
  console.log(`Retired watchlist defaults removed: ${removed.count}`);
}

// Re-decode titles/descriptions stored before the entity fix (&#8217; etc.).
const articles = await prisma.article.findMany({ select: { id: true, title: true, description: true, author: true } });
let fixed = 0;
for (const a of articles) {
  const title = stripHtml(a.title);
  const description = a.description ? stripHtml(a.description) : null;
  const author = a.author ? stripHtml(a.author) : null;
  if (title !== a.title || description !== a.description || author !== a.author) {
    await prisma.article.update({ where: { id: a.id }, data: { title, description, author } });
    fixed += 1;
  }
}
console.log(`Re-decoded ${fixed}/${articles.length} articles`);

const relink = await relinkAllWatchlist();
console.log(`Re-linked watchlist: ${relink.links} links (${relink.items} items, ${relink.articles} articles)`);

const themes = await extractThemesFromDb({ windowHours: 24 });
console.log(`Themes refreshed: ${themes.length}`);
for (const t of themes.slice(0, 10)) {
  console.log(`  ${t.name} (${t.articleCount}, vel ${t.velocity}, ${t.sentimentLabel})`);
}

// Verify the daily crossword builds from current news.
const puzzle = await getDailyCrossword({ date: new Date().toISOString().slice(0, 10) });
console.log(`Crossword: ${puzzle.wordCount} words, ${Object.keys(puzzle.cells).length} cells, clues=${puzzle.across.length + puzzle.down.length}`);

await prisma.$disconnect();
