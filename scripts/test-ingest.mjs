// Quick end-to-end smoke test for the Market Signal data pipeline.
// Run: node --env-file=.env scripts/test-ingest.mjs
import prisma from '../lib/db.js';
import { ingestAll } from '../lib/ingest.js';
import { extractThemesFromDb } from '../lib/themes.js';
import { getTopStories, getWatchlistWithStories, getDigest } from '../lib/signal.js';
import { relinkAllWatchlist } from '../lib/watchlist.js';

const result = await ingestAll();
console.log('INGEST STATS:', JSON.stringify(result.stats, null, 2));

const count = await prisma.article.count();
console.log(`\nTOTAL ARTICLES IN DB: ${count}`);

const top = await getTopStories({ limit: 3 });
console.log('\nTOP 3 STORIES:');
for (const a of top) {
  console.log(`  [${a.sentimentLabel}] (${a.reliabilityScore}) ${a.title} — ${a.source?.name} | summary: ${(a.summary || '').slice(0, 80)}`);
}

const themes = await extractThemesFromDb({ windowHours: 24 });
console.log(`\nTHEMES (${themes.length}):`);
for (const t of themes.slice(0, 8)) {
  console.log(`  ${t.name} (${t.articleCount} articles, vel ${t.velocity}, ${t.sentimentLabel})`);
}

const relink = await relinkAllWatchlist();
console.log(`\nRE-LINK: ${relink.links} links across ${relink.items} items / ${relink.articles} articles`);

const watch = await getWatchlistWithStories({ storiesPerItem: 2 });
console.log('\nWATCHLIST (after relink):');
for (const w of watch) {
  console.log(`  ${w.name}: ${w.articleCount} stories${w.stories.length ? ` | e.g. "${w.stories[0].title.slice(0, 60)}"` : ''}`);
}

const digest = await getDigest();
console.log(`\nDIGEST: themes=${digest.themes.length}, stories=${digest.stories.length}, watchlist=${digest.watchlist.length}`);

await prisma.$disconnect();
