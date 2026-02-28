/**
 * News source configurations and topic category definitions.
 */

const NEWS_SOURCES = {
  rss: [
    { name: 'BBC News', url: 'http://feeds.bbci.co.uk/news/rss.xml', category: 'general' },
    { name: 'Reuters', url: 'https://feeds.reuters.com/reuters/topNews', category: 'general' },
    { name: 'TechCrunch', url: 'https://techcrunch.com/feed/', category: 'technology' },
    { name: 'CNN', url: 'http://rss.cnn.com/rss/edition.rss', category: 'general' },
    { name: 'The Guardian', url: 'https://www.theguardian.com/world/rss', category: 'world' },
    { name: 'Hacker News', url: 'https://hnrss.org/frontpage', category: 'technology' },
    { name: 'Financial Times', url: 'https://www.ft.com/rss/home', category: 'business' },
    { name: 'NPR', url: 'https://feeds.npr.org/1001/rss.xml', category: 'general' }
  ],
  apis: [
    {
      name: 'NewsAPI',
      endpoint: 'https://newsapi.org/v2/top-headlines',
      key: process.env.NEWS_API_KEY,
      params: { country: 'us', pageSize: 50 }
    },
    {
      name: 'Google News',
      endpoint: 'https://www.googleapis.com/customsearch/v1',
      key: process.env.NEXT_PUBLIC_NEWS_API_KEY,
      params: { cx: process.env.GOOGLE_CSE_ID, num: 10 }
    }
  ]
};

const TOPIC_CATEGORIES = {
  technology: ['tech', 'ai', 'software', 'startup', 'innovation', 'digital', 'cyber'],
  business: ['finance', 'economy', 'market', 'stock', 'investment', 'corporate', 'trade'],
  politics: ['government', 'election', 'policy', 'congress', 'senate', 'president', 'law'],
  health: ['medical', 'healthcare', 'disease', 'vaccine', 'hospital', 'doctor', 'medicine'],
  science: ['research', 'study', 'discovery', 'experiment', 'scientist', 'climate', 'space'],
  sports: ['football', 'basketball', 'soccer', 'baseball', 'olympics', 'championship', 'athlete'],
  entertainment: ['movie', 'music', 'celebrity', 'hollywood', 'tv', 'streaming', 'gaming'],
  world: ['international', 'global', 'country', 'war', 'conflict', 'diplomacy', 'foreign']
};

module.exports = { NEWS_SOURCES, TOPIC_CATEGORIES };
