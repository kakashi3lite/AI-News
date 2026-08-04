/**
 * Curated RSS sources for the Market Signal dashboard.
 *
 * reliabilityScore (0–1) reflects general editorial rigor / verification
 * standards, and is shown on every story so users can judge accuracy.
 * No API keys required — RSS feeds are the guaranteed "real data" path.
 */

export const RSS_SOURCES = [
  { name: 'Wall Street Journal', url: 'https://feeds.a.dj.com/rss/RSSMarketsMain.xml', category: 'business', reliability: 0.96 },
  { name: 'Bloomberg', url: 'https://feeds.bloomberg.com/markets/news.rss', category: 'business', reliability: 0.96 },
  { name: 'Financial Times', url: 'https://www.ft.com/rss/home', category: 'business', reliability: 0.95 },
  { name: 'BBC News', url: 'http://feeds.bbci.co.uk/news/rss.xml', category: 'general', reliability: 0.94 },
  { name: 'NPR', url: 'https://feeds.npr.org/1001/rss.xml', category: 'general', reliability: 0.93 },
  { name: 'The Guardian', url: 'https://www.theguardian.com/world/rss', category: 'world', reliability: 0.92 },
  { name: 'CNN', url: 'http://rss.cnn.com/rss/edition.rss', category: 'general', reliability: 0.88 },
  { name: 'The Verge', url: 'https://www.theverge.com/rss/index.xml', category: 'technology', reliability: 0.87 },
  { name: 'TechCrunch', url: 'https://techcrunch.com/feed/', category: 'technology', reliability: 0.85 },
  { name: 'Hacker News', url: 'https://hnrss.org/frontpage', category: 'technology', reliability: 0.8 },
  { name: 'Google News', url: 'https://news.google.com/rss?hl=en-US&gl=US&ceid=US:en', category: 'general', reliability: 0.9 },
  { name: 'Google News Business', url: 'https://news.google.com/rss/headlines/section/topic/BUSINESS?hl=en-US&gl=US&ceid=US:en', category: 'business', reliability: 0.9 },
  { name: 'Google News Tech', url: 'https://news.google.com/rss/headlines/section/topic/TECHNOLOGY?hl=en-US&gl=US&ceid=US:en', category: 'technology', reliability: 0.9 },
];

// Optional API sources — used only when the relevant key is present.
export const API_SOURCES = [
  {
    name: 'NewsAPI',
    endpoint: 'https://newsapi.org/v2/top-headlines',
    key: () => process.env.NEWS_API_KEY,
    params: { country: 'us', pageSize: 30 },
  },
];

export const TOPIC_CATEGORIES = {
  technology: ['tech', 'ai', 'artificial intelligence', 'software', 'startup', 'innovation', 'digital', 'cyber', 'chip', 'semiconductor', 'cloud', 'gpu', 'robot'],
  business: ['finance', 'economy', 'market', 'stock', 'investment', 'corporate', 'trade', 'earnings', 'revenue', 'acquisition', 'merger', 'ipo'],
  politics: ['government', 'election', 'policy', 'congress', 'senate', 'president', 'law', 'legislation', 'regulation'],
  health: ['medical', 'healthcare', 'disease', 'vaccine', 'hospital', 'doctor', 'medicine', 'drug'],
  science: ['research', 'study', 'discovery', 'experiment', 'scientist', 'climate', 'space', 'physics', 'nasa'],
  world: ['international', 'global', 'war', 'conflict', 'diplomacy', 'foreign', 'sanction'],
};

/** Classify an article into one of the TOPIC_CATEGORIES by keyword frequency. */
export function classifyArticle(title, description = '') {
  const text = `${title} ${description}`.toLowerCase();
  let best = { category: 'general', score: 0 };
  for (const [category, keywords] of Object.entries(TOPIC_CATEGORIES)) {
    let score = 0;
    for (const k of keywords) {
      const re = new RegExp(`\\b${k.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}\\b`, 'g');
      const matches = text.match(re);
      if (matches) score += matches.length;
    }
    if (score > best.score) best = { category, score };
  }
  return best.category;
}
