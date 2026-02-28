/**
 * News Ingestion Orchestrator
 * Coordinates feed parsing, normalization, deduplication, and classification.
 */

const fs = require('fs').promises;
const path = require('path');
const { errorHandler } = require('../lib/errorHandler');
const { NEWS_SOURCES } = require('./sources');
const { fetchRSSFeed, fetchAPISource } = require('./feedParser');
const { normalizeRSSArticle, normalizeNewsAPIArticle, normalizeGoogleNewsArticle } = require('./articleNormalizer');
const { deduplicateArticles } = require('./articleDeduplicator');
const { classifyTopicsTransformer, basicTopicClassification } = require('./topicClassifier');

const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

const chunkArray = (array, size) => {
  const chunks = [];
  for (let i = 0; i < array.length; i += size) {
    chunks.push(array.slice(i, i + size));
  }
  return chunks;
};

class NewsIngestionEngine {
  constructor(options = {}) {
    this.cacheDir = options.cacheDir || path.join(__dirname, '../cache');
    this.maxConcurrent = options.maxConcurrent || 5;
    this.rateLimitDelay = options.rateLimitDelay || 1000;
    this.lastFetch = new Map();
    this.initializeCache();
  }

  async initializeCache() {
    await errorHandler.executeWithErrorHandling(
      async () => {
        await fs.mkdir(this.cacheDir, { recursive: true });
        console.log(`📁 Cache directory initialized: ${this.cacheDir}`);
      },
      {
        name: 'cache_initialization',
        retryOptions: { maxRetries: 2 },
        fallback: () => {
          console.warn('⚠️ Using temporary cache directory');
          this.cacheDir = path.join(require('os').tmpdir(), 'news-cache');
        }
      }
    );
  }

  async ingestAllSources(options = {}) {
    console.log('🚀 Starting news ingestion from all sources...');
    const startTime = Date.now();
    const results = { articles: [], sources: [], errors: [], stats: { totalFetched: 0, duplicatesRemoved: 0, categorized: 0, processingTime: 0 } };

    try {
      const rssResults = await errorHandler.executeWithErrorHandling(
        () => this.ingestRSSSources(options),
        { name: 'rss_ingestion', retryOptions: { maxRetries: 2 }, fallback: () => ({ articles: [], sources: [], errors: [{ source: 'rss', error: 'RSS ingestion failed' }] }) }
      );
      results.articles.push(...rssResults.articles);
      results.sources.push(...rssResults.sources);
      results.errors.push(...rssResults.errors);

      const apiResults = await errorHandler.executeWithErrorHandling(
        () => this.ingestAPISources(options),
        { name: 'api_ingestion', retryOptions: { maxRetries: 2 }, fallback: () => ({ articles: [], sources: [], errors: [{ source: 'api', error: 'API ingestion failed' }] }) }
      );
      results.articles.push(...apiResults.articles);
      results.sources.push(...apiResults.sources);
      results.errors.push(...apiResults.errors);

      const uniqueArticles = deduplicateArticles(results.articles);
      results.stats.duplicatesRemoved = results.articles.length - uniqueArticles.length;
      results.articles = uniqueArticles;

      results.articles = await errorHandler.executeWithErrorHandling(
        () => classifyTopicsTransformer(results.articles, this.cacheDir),
        { name: 'topic_classification', retryOptions: { maxRetries: 1 }, fallback: () => basicTopicClassification(results.articles) }
      );
      results.stats.categorized = results.articles.length;

      await errorHandler.executeWithErrorHandling(
        () => this.cacheResults(results.articles),
        { name: 'cache_results', retryOptions: { maxRetries: 1 }, fallback: () => console.warn('⚠️ Failed to cache results') }
      );

      results.stats.totalFetched = results.articles.length;
      results.stats.processingTime = Date.now() - startTime;
      console.log(`✅ Ingestion complete: ${results.stats.totalFetched} articles in ${results.stats.processingTime}ms`);
    } catch (error) {
      console.error('❌ Ingestion failed:', error);
      results.errors.push({ source: 'main', error: error.message });

      if (results.articles.length === 0) {
        try {
          const fallback = await this.emergencyFallback();
          results.articles.push(...fallback.articles);
          results.sources.push(...fallback.sources);
        } catch (fallbackError) {
          console.error('❌ Emergency fallback also failed:', fallbackError.message);
        }
      }
    }

    return results;
  }

  async ingestRSSSources(options = {}) {
    const results = { articles: [], sources: [], errors: [] };
    const sources = options.sources || NEWS_SOURCES.rss;
    console.log(`📡 Fetching from ${sources.length} RSS sources...`);

    const chunks = chunkArray(sources, this.maxConcurrent);
    for (const chunk of chunks) {
      const chunkResults = await Promise.allSettled(
        chunk.map((source) => this.fetchSingleRSSSource(source))
      );
      chunkResults.forEach((result, i) => {
        const source = chunk[i];
        if (result.status === 'fulfilled') {
          results.articles.push(...result.value.articles);
          results.sources.push({ name: source.name, status: 'success', count: result.value.articles.length });
        } else {
          results.errors.push({ source: source.name, error: result.reason.message });
          results.sources.push({ name: source.name, status: 'error', error: result.reason.message });
        }
      });
      if (chunks.indexOf(chunk) < chunks.length - 1) await delay(this.rateLimitDelay);
    }
    return results;
  }

  async fetchSingleRSSSource(source) {
    const cacheKey = `rss_${source.name.replace(/\s+/g, '_').toLowerCase()}`;
    const lastFetch = this.lastFetch.get(cacheKey) || 0;
    if (Date.now() - lastFetch < 15 * 60 * 1000) {
      console.log(`⏭️  Skipping ${source.name} (recently fetched)`);
      return { articles: [] };
    }

    try {
      console.log(`📰 Fetching RSS: ${source.name}`);
      const items = await fetchRSSFeed(source.url);
      const articles = items.map((item) => normalizeRSSArticle(item, source));
      this.lastFetch.set(cacheKey, Date.now());
      console.log(`✅ ${source.name}: ${articles.length} articles`);
      return { articles };
    } catch (error) {
      console.error(`❌ Failed to fetch RSS from ${source.name}:`, error.message);
      const cached = await this.getCachedRSSData(source);
      if (cached?.articles) return cached;
      throw error;
    }
  }

  async ingestAPISources(options = {}) {
    const results = { articles: [], sources: [], errors: [] };
    const sources = options.sources || NEWS_SOURCES.apis;
    console.log(`🔌 Fetching from ${sources.length} API sources...`);

    for (const source of sources) {
      if (!source.key) {
        console.warn(`⚠️  Skipping ${source.name}: No API key configured`);
        continue;
      }
      try {
        const data = await fetchAPISource(source, options);
        let articles = [];
        if (source.name === 'NewsAPI' && data.articles) {
          articles = data.articles.map((item) => normalizeNewsAPIArticle(item, source));
        } else if (source.name === 'Google News' && data.items) {
          articles = data.items.map((item) => normalizeGoogleNewsArticle(item, source));
        }
        results.articles.push(...articles);
        results.sources.push({ name: source.name, status: 'success', count: articles.length });
        console.log(`✅ ${source.name}: ${articles.length} articles`);
      } catch (error) {
        console.error(`❌ API fetch failed for ${source.name}:`, error.message);
        results.errors.push({ source: source.name, error: error.message });
        results.sources.push({ name: source.name, status: 'error', error: error.message });
      }
      await delay(this.rateLimitDelay);
    }
    return results;
  }

  async getCachedRSSData(source) {
    try {
      const cacheKey = `rss_cache_${source.name.replace(/\s+/g, '_').toLowerCase()}`;
      const cacheFile = path.join(this.cacheDir, `${cacheKey}.json`);
      const cached = JSON.parse(await fs.readFile(cacheFile, 'utf8'));
      if (cached?.timestamp > Date.now() - 24 * 60 * 60 * 1000) return cached.data;
      return null;
    } catch {
      return null;
    }
  }

  async emergencyFallback() {
    console.log('🚨 Attempting emergency fallback...');
    const fallbackSources = [
      { name: 'BBC News', url: 'http://feeds.bbci.co.uk/news/rss.xml', category: 'general' },
      { name: 'Reuters', url: 'https://feeds.reuters.com/reuters/topNews', category: 'general' }
    ];
    for (const source of fallbackSources) {
      try {
        const items = await fetchRSSFeed(source.url, 5000);
        const articles = items.slice(0, 10).map((item) => normalizeRSSArticle(item, source));
        if (articles.length > 0) {
          console.log(`✅ Emergency fallback successful with ${source.name}`);
          return { articles, sources: [{ name: source.name, status: 'emergency_fallback' }] };
        }
      } catch {
        console.log(`❌ Emergency fallback failed for ${source.name}`);
      }
    }
    return {
      articles: [{ id: 'emergency_' + Date.now(), title: 'News Service Temporarily Unavailable', description: 'Our news ingestion service is experiencing temporary issues. Please check back later.', content: 'Our news ingestion service is experiencing temporary issues. Please check back later.', url: '#', image: '', publishedAt: new Date().toISOString(), source: { name: 'System', url: '#' }, category: 'system', tags: ['system'], ingestionMethod: 'emergency', fetchedAt: new Date().toISOString() }],
      sources: [{ name: 'Emergency', status: 'fallback' }]
    };
  }

  async cacheResults(articles) {
    const cacheFile = path.join(this.cacheDir, `articles_${Date.now()}.json`);
    await fs.writeFile(cacheFile, JSON.stringify({ timestamp: new Date().toISOString(), count: articles.length, articles }, null, 2));
    console.log(`💾 Cached ${articles.length} articles to ${cacheFile}`);
  }
}

module.exports = { NewsIngestionEngine, NEWS_SOURCES: require('./sources').NEWS_SOURCES, TOPIC_CATEGORIES: require('./sources').TOPIC_CATEGORIES };

if (require.main === module) {
  const engine = new NewsIngestionEngine();
  engine.ingestAllSources({ query: process.argv[2], category: process.argv[3] }).then((results) => {
    console.log(`\n📊 Ingestion Results:`);
    console.log(`Total articles: ${results.stats.totalFetched}`);
    console.log(`Duplicates removed: ${results.stats.duplicatesRemoved}`);
    console.log(`Processing time: ${results.stats.processingTime}ms`);
    console.log(`Errors: ${results.errors.length}`);
    if (results.errors.length > 0) {
      console.log('\n❌ Errors:');
      results.errors.forEach((err) => console.log(`  - ${err.source}: ${err.error}`));
    }
  }).catch(console.error);
}
