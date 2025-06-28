/**
 * Enhanced News Ingestion Module
 * Part of Dr. NewsForge's AI News Dashboard
 * 
 * Features:
 * - Async RSS/API crawling with adaptive scheduling
 * - Topic classification via transformer-based models
 * - Multi-source aggregation (RSS, APIs, web scraping)
 * - Rate limiting and error handling
 * - Caching and deduplication
 */

const axios = require('axios');
const { XMLParser } = require('fast-xml-parser');
const crypto = require('crypto');
const fs = require('fs').promises;
const path = require('path');
const { spawn } = require('child_process');
const { errorHandler, ERROR_TYPES } = require('../lib/errorHandler');

// News source configurations
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

// Topic classification categories
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

class NewsIngestionEngine {
  constructor(options = {}) {
    this.cacheDir = options.cacheDir || path.join(__dirname, '../cache');
    this.maxConcurrent = options.maxConcurrent || 5;
    this.rateLimitDelay = options.rateLimitDelay || 1000;
    this.deduplicationWindow = options.deduplicationWindow || 24 * 60 * 60 * 1000; // 24 hours
    this.xmlParser = new XMLParser({
      ignoreAttributes: false,
      attributeNamePrefix: '@_'
    });
    this.articleCache = new Map();
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

  /**
   * Main ingestion method - fetches from all sources
   */
  async ingestAllSources(options = {}) {
    console.log('🚀 Starting news ingestion from all sources...');
    const startTime = Date.now();
    
    const results = {
      articles: [],
      sources: [],
      errors: [],
      stats: {
        totalFetched: 0,
        duplicatesRemoved: 0,
        categorized: 0,
        processingTime: 0
      }
    };

    try {
      // Fetch from RSS sources with error handling
      const rssResults = await errorHandler.executeWithErrorHandling(
        () => this.ingestRSSSources(options),
        {
          name: 'rss_ingestion',
          retryOptions: { maxRetries: 2 },
          fallback: () => ({ articles: [], sources: [], errors: [{ source: 'rss', error: 'RSS ingestion failed, using fallback' }] })
        }
      );
      results.articles.push(...rssResults.articles);
      results.sources.push(...rssResults.sources);
      results.errors.push(...rssResults.errors);

      // Fetch from API sources with error handling
      const apiResults = await errorHandler.executeWithErrorHandling(
        () => this.ingestAPISources(options),
        {
          name: 'api_ingestion',
          retryOptions: { maxRetries: 2 },
          fallback: () => ({ articles: [], sources: [], errors: [{ source: 'api', error: 'API ingestion failed, using fallback' }] })
        }
      );
      results.articles.push(...apiResults.articles);
      results.sources.push(...apiResults.sources);
      results.errors.push(...apiResults.errors);

      // Deduplicate articles
      const uniqueArticles = this.deduplicateArticles(results.articles);
      results.stats.duplicatesRemoved = results.articles.length - uniqueArticles.length;
      results.articles = uniqueArticles;

      // Classify topics with error handling
      results.articles = await errorHandler.executeWithErrorHandling(
        () => this.classifyTopics(results.articles),
        {
          name: 'topic_classification',
          retryOptions: { maxRetries: 1 },
          fallback: () => {
            console.warn('⚠️ Topic classification failed, using basic categorization');
            return this.basicTopicClassification(results.articles);
          }
        }
      );
      results.stats.categorized = results.articles.length;

      // Cache results with error handling
      await errorHandler.executeWithErrorHandling(
        () => this.cacheResults(results.articles),
        {
          name: 'cache_results',
          retryOptions: { maxRetries: 1 },
          fallback: () => console.warn('⚠️ Failed to cache results, continuing without caching')
        }
      );

      results.stats.totalFetched = results.articles.length;
      results.stats.processingTime = Date.now() - startTime;

      console.log(`✅ Ingestion complete: ${results.stats.totalFetched} articles in ${results.stats.processingTime}ms`);
      
    } catch (error) {
      console.error('❌ Ingestion failed:', error);
      results.errors.push({ source: 'main', error: error.message });
      
      // Attempt graceful degradation
      if (results.articles.length === 0) {
        console.log('🔄 Attempting emergency fallback...');
        try {
          const fallbackResults = await this.emergencyFallback();
          results.articles.push(...fallbackResults.articles);
          results.sources.push(...fallbackResults.sources);
        } catch (fallbackError) {
          console.error('❌ Emergency fallback also failed:', fallbackError.message);
        }
      }
    }

    return results;
  }

  /**
   * Fetch articles from RSS sources
   */
  async ingestRSSSources(options = {}) {
    const results = { articles: [], sources: [], errors: [] };
    const sources = options.sources || NEWS_SOURCES.rss;
    
    console.log(`📡 Fetching from ${sources.length} RSS sources...`);

    // Process sources with concurrency control
    const chunks = this.chunkArray(sources, this.maxConcurrent);
    
    for (const chunk of chunks) {
      const promises = chunk.map(source => this.fetchRSSSource(source));
      const chunkResults = await Promise.allSettled(promises);
      
      chunkResults.forEach((result, index) => {
        const source = chunk[index];
        if (result.status === 'fulfilled') {
          results.articles.push(...result.value.articles);
          results.sources.push({ name: source.name, status: 'success', count: result.value.articles.length });
        } else {
          console.error(`❌ Failed to fetch ${source.name}:`, result.reason);
          results.errors.push({ source: source.name, error: result.reason.message });
          results.sources.push({ name: source.name, status: 'error', error: result.reason.message });
        }
      });
      
      // Rate limiting between chunks
      if (chunks.indexOf(chunk) < chunks.length - 1) {
        await this.delay(this.rateLimitDelay);
      }
    }

    return results;
  }

  /**
   * Fetch articles from a single RSS source
   */
  async fetchRSSSource(source) {
    const cacheKey = `rss_${source.name.replace(/\s+/g, '_').toLowerCase()}`;
    const lastFetch = this.lastFetch.get(cacheKey) || 0;
    const now = Date.now();
    
    // Skip if fetched recently (adaptive scheduling)
    if (now - lastFetch < 15 * 60 * 1000) { // 15 minutes
      console.log(`⏭️  Skipping ${source.name} (recently fetched)`);
      return { articles: [] };
    }

    try {
      console.log(`📰 Fetching RSS: ${source.name}`);
      
      const response = await axios.get(source.url, {
        timeout: 10000,
        headers: {
          'User-Agent': 'NewsForge-AI-Dashboard/1.0 (https://github.com/newsforge/ai-dashboard)'
        }
      });

      const parsed = this.xmlParser.parse(response.data);
      const items = this.extractRSSItems(parsed);
      
      const articles = items.map((item, index) => this.normalizeRSSArticle(item, source, index));
      
      this.lastFetch.set(cacheKey, now);
      console.log(`✅ ${source.name}: ${articles.length} articles`);
      
      return { articles };
    } catch (error) {
      console.error(`❌ Failed to fetch RSS from ${source.name}:`, error.message);
      
      // Try to get cached data as fallback
      const cachedData = await this.getCachedRSSData(source);
      if (cachedData && cachedData.articles) {
        console.log(`🔄 Using cached data for ${source.name}`);
        return cachedData;
      }
      
      throw error;
    }
  }

  /**
   * Extract items from parsed RSS/XML
   */
  extractRSSItems(parsed) {
    // Handle different RSS formats
    if (parsed.rss?.channel?.item) {
      return Array.isArray(parsed.rss.channel.item) ? parsed.rss.channel.item : [parsed.rss.channel.item];
    }
    if (parsed.feed?.entry) {
      return Array.isArray(parsed.feed.entry) ? parsed.feed.entry : [parsed.feed.entry];
    }
    if (parsed.channel?.item) {
      return Array.isArray(parsed.channel.item) ? parsed.channel.item : [parsed.channel.item];
    }
    return [];
  }

  /**
   * Normalize RSS article to standard format
   */
  normalizeRSSArticle(item, source, index) {
    const title = item.title || item.summary || 'No title';
    const description = item.description || item.summary || item.content || '';
    const link = item.link?.['@_href'] || item.link || item.guid || '';
    const pubDate = item.pubDate || item.published || item.updated || new Date().toISOString();
    
    // Generate unique ID
    const id = this.generateArticleId(title, link, source.name);
    
    return {
      id,
      title: this.cleanText(title),
      description: this.cleanText(description),
      content: this.cleanText(description),
      url: link,
      image: this.extractImageUrl(item),
      publishedAt: this.normalizeDate(pubDate),
      source: {
        name: source.name,
        url: source.url
      },
      category: source.category || 'general',
      tags: this.extractTags(item),
      ingestionMethod: 'rss',
      fetchedAt: new Date().toISOString()
    };
  }

  /**
   * Fetch articles from API sources
   */
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
        const sourceResults = await this.fetchAPISource(source, options);
        results.articles.push(...sourceResults.articles);
        results.sources.push({ name: source.name, status: 'success', count: sourceResults.articles.length });
      } catch (error) {
        console.error(`❌ API fetch failed for ${source.name}:`, error.message);
        results.errors.push({ source: source.name, error: error.message });
        results.sources.push({ name: source.name, status: 'error', error: error.message });
      }
      
      await this.delay(this.rateLimitDelay);
    }

    return results;
  }

  /**
   * Fetch from a single API source
   */
  async fetchAPISource(source, options = {}) {
    console.log(`🔌 Fetching API: ${source.name}`);
    
    let params = { ...source.params };
    
    // Add query parameters if provided
    if (options.query) {
      if (source.name === 'NewsAPI') {
        params.q = options.query;
      } else if (source.name === 'Google News') {
        params.q = options.query;
      }
    }
    
    if (options.category && source.name === 'NewsAPI') {
      params.category = options.category;
    }

    const response = await axios.get(source.endpoint, {
      params: { ...params, key: source.key },
      timeout: 15000,
      headers: {
        'User-Agent': 'NewsForge-AI-Dashboard/1.0'
      }
    });

    let articles = [];
    
    if (source.name === 'NewsAPI' && response.data.articles) {
      articles = response.data.articles.map((item, index) => 
        this.normalizeNewsAPIArticle(item, source, index)
      );
    } else if (source.name === 'Google News' && response.data.items) {
      articles = response.data.items.map((item, index) => 
        this.normalizeGoogleNewsArticle(item, source, index)
      );
    }

    console.log(`✅ ${source.name}: ${articles.length} articles`);
    return { articles };
  }

  /**
   * Get cached RSS data as fallback
   */
  async getCachedRSSData(source) {
    try {
      const cacheKey = `rss_cache_${source.name.replace(/\s+/g, '_').toLowerCase()}`;
      const cacheFile = path.join(this.cacheDir, `${cacheKey}.json`);
      const cached = JSON.parse(await fs.readFile(cacheFile, 'utf8'));
      if (cached && cached.timestamp > Date.now() - 24 * 60 * 60 * 1000) { // 24 hours
        console.log(`📦 Retrieved cached data for ${source.name}`);
        return cached.data;
      }
      return null;
    } catch (error) {
      console.error(`❌ Failed to get cached data for ${source.name}:`, error.message);
      return null;
    }
  }

  /**
   * Emergency fallback for complete ingestion failure
   */
  async emergencyFallback() {
    try {
      console.log('🚨 Attempting emergency fallback...');
      const fallbackSources = [
        { name: 'BBC News', url: 'http://feeds.bbci.co.uk/news/rss.xml', category: 'general' },
        { name: 'Reuters', url: 'https://feeds.reuters.com/reuters/topNews', category: 'general' }
      ];
      
      for (const source of fallbackSources) {
        try {
          const response = await axios.get(source.url, { timeout: 5000 });
          const parsed = this.xmlParser.parse(response.data);
          const items = this.extractRSSItems(parsed);
          const articles = items.slice(0, 10).map((item, index) => this.normalizeRSSArticle(item, source, index));
          if (articles.length > 0) {
            console.log(`✅ Emergency fallback successful with ${source.name}`);
            return { articles, sources: [{ name: source.name, status: 'emergency_fallback' }] };
          }
        } catch (error) {
          console.log(`❌ Emergency fallback failed for ${source.name}`);
        }
      }
      
      // Return minimal default content
      return {
        articles: [{
          id: 'emergency_' + Date.now(),
          title: 'News Service Temporarily Unavailable',
          description: 'Our news ingestion service is experiencing temporary issues. Please check back later.',
          content: 'Our news ingestion service is experiencing temporary issues. Please check back later.',
          url: '#',
          image: '',
          publishedAt: new Date().toISOString(),
          source: { name: 'System', url: '#' },
          category: 'system',
          tags: ['system'],
          ingestionMethod: 'emergency',
          fetchedAt: new Date().toISOString()
        }],
        sources: [{ name: 'Emergency', status: 'fallback' }]
      };
    } catch (error) {
      console.error('❌ Emergency fallback completely failed:', error.message);
      return { articles: [], sources: [] };
    }
  }

  /**
   * Basic topic classification fallback
   */
  basicTopicClassification(articles) {
    console.log(`🏷️  Using basic topic classification for ${articles.length} articles...`);
    
    return articles.map(article => {
      const text = `${article.title} ${article.description}`.toLowerCase();
      
      // Simple keyword-based classification
      if (text.includes('tech') || text.includes('ai') || text.includes('software')) {
        article.category = 'technology';
      } else if (text.includes('business') || text.includes('finance') || text.includes('market')) {
        article.category = 'business';
      } else if (text.includes('health') || text.includes('medical')) {
        article.category = 'health';
      } else if (text.includes('sport') || text.includes('game')) {
        article.category = 'sports';
      }
      
      return article;
    });
  }

  /**
   * Normalize NewsAPI article
   */
  normalizeNewsAPIArticle(item, source, index) {
    const id = this.generateArticleId(item.title, item.url, source.name);
    
    return {
      id,
      title: this.cleanText(item.title || 'No title'),
      description: this.cleanText(item.description || ''),
      content: this.cleanText(item.content || item.description || ''),
      url: item.url || '',
      image: item.urlToImage || '',
      publishedAt: this.normalizeDate(item.publishedAt),
      source: {
        name: item.source?.name || source.name,
        url: item.url
      },
      category: 'general',
      tags: [],
      ingestionMethod: 'api-newsapi',
      fetchedAt: new Date().toISOString()
    };
  }

  /**
   * Normalize Google News article
   */
  normalizeGoogleNewsArticle(item, source, index) {
    const id = this.generateArticleId(item.title, item.link, source.name);
    
    return {
      id,
      title: this.cleanText(item.title || 'No title'),
      description: this.cleanText(item.snippet || ''),
      content: this.cleanText(item.snippet || ''),
      url: item.link || '',
      image: item.pagemap?.cse_image?.[0]?.src || '',
      publishedAt: new Date().toISOString(),
      source: {
        name: item.displayLink || source.name,
        url: item.link
      },
      category: 'general',
      tags: item.pagemap?.metatags?.[0]?.news_keywords?.split(',').map(t => t.trim()).filter(Boolean) || [],
      ingestionMethod: 'api-google',
      fetchedAt: new Date().toISOString()
    };
  }

  /**
   * Enhanced classification with transformer fallback
   */
  async classifyTopics(articles) {
    console.log(`🏷️  Classifying topics for ${articles.length} articles...`);
    
    try {
      // Try transformer-based classification first
      // Check if transformer classifier is available
      const transformerPath = path.join(__dirname, 'transformer_classifier.py');
      const fsSync = require('fs');
      
      if (fsSync.existsSync(transformerPath)) {
        console.log('🤖 Using transformer-based classification...');
        
        // Prepare articles for Python classifier
        const articlesData = articles.map(article => ({
          title: article.title,
          description: article.description,
          content: article.content,
          source: article.source,
          url: article.url
        }));
        
        // Create temporary file for secure data transfer
        const tempFile = path.join(this.cacheDir, `articles_${Date.now()}.json`);
        await fs.writeFile(tempFile, JSON.stringify(articlesData));
        
        // Call Python transformer classifier with file input
        const pythonScript = `
import asyncio
import json
import sys
import os
sys.path.append('${__dirname.replace(/\\/g, '/')}')
from transformer_classifier import classify_topics_transformer

async def main():
    with open('${tempFile.replace(/\\/g, '/')}', 'r', encoding='utf-8') as f:
        articles = json.load(f)
    results = await classify_topics_transformer(articles)
    print(json.dumps(results))
    os.unlink('${tempFile.replace(/\\/g, '/')}')

asyncio.run(main())
        `;
        
        const pythonProcess = spawn('python', ['-c', pythonScript]);
        
        let output = '';
        let error = '';
        
        pythonProcess.stdout.on('data', (data) => {
          output += data.toString();
        });
        
        pythonProcess.stderr.on('data', (data) => {
          error += data.toString();
        });
        
        return new Promise((resolve) => {
          pythonProcess.on('close', async (code) => {
            // Cleanup temporary file if it still exists
            try {
              await fs.unlink(tempFile);
            } catch (cleanupError) {
              // File might already be deleted by Python script
            }
            
            if (code === 0 && output.trim()) {
              try {
                const classifiedArticles = JSON.parse(output.trim());
                console.log('✅ Transformer classification completed successfully');
                resolve(classifiedArticles);
              } catch (parseError) {
                console.warn('⚠️ Failed to parse transformer results, falling back to keyword classification');
                resolve(this.classifyTopicsKeyword(articles));
              }
            } else {
              console.warn(`⚠️ Transformer classification failed (code: ${code}), falling back to keyword classification`);
              if (error) console.warn('Error:', error);
              resolve(this.classifyTopicsKeyword(articles));
            }
          });
        });
      } else {
        console.log('📝 Transformer classifier not found, using keyword classification');
        return this.classifyTopicsKeyword(articles);
      }
    } catch (error) {
      console.warn('⚠️ Error in transformer classification, falling back to keyword classification:', error.message);
      return this.classifyTopicsKeyword(articles);
    }
  }

  /**
   * Keyword-based classification fallback
   */
  classifyTopicsKeyword(articles) {
    console.log(`🏷️  Using keyword-based classification for ${articles.length} articles...`);
    
    return articles.map(article => {
      const text = `${article.title} ${article.description}`.toLowerCase();
      const scores = {};
      
      // Calculate category scores
      Object.entries(TOPIC_CATEGORIES).forEach(([category, keywords]) => {
        scores[category] = keywords.reduce((score, keyword) => {
          const matches = (text.match(new RegExp(keyword, 'g')) || []).length;
          return score + matches;
        }, 0);
      });
      
      // Find best category
      const bestCategory = Object.entries(scores).reduce((best, [category, score]) => 
        score > best.score ? { category, score } : best,
        { category: 'general', score: 0 }
      );
      
      // Update article category if confidence is high enough
      if (bestCategory.score > 0) {
        article.category = bestCategory.category;
        article.topicConfidence = bestCategory.score;
      }
      
      article.classificationMethod = 'keyword';
      return article;
    });
  }

  /**
   * Remove duplicate articles based on title similarity and URL
   */
  deduplicateArticles(articles) {
    console.log(`🔄 Deduplicating ${articles.length} articles...`);
    
    const seen = new Set();
    const unique = [];
    
    for (const article of articles) {
      const fingerprint = this.generateFingerprint(article);
      
      if (!seen.has(fingerprint)) {
        seen.add(fingerprint);
        unique.push(article);
      }
    }
    
    console.log(`✅ Removed ${articles.length - unique.length} duplicates`);
    return unique;
  }

  /**
   * Generate article fingerprint for deduplication
   */
  generateFingerprint(article) {
    const titleWords = article.title.toLowerCase()
      .replace(/[^a-z0-9\s]/g, '')
      .split(/\s+/)
      .filter(word => word.length > 3)
      .slice(0, 5)
      .sort()
      .join(' ');
    
    let urlDomain = '';
    if (article.url) {
      try {
        urlDomain = new URL(article.url).hostname;
      } catch (error) {
        // Handle invalid URLs gracefully
        urlDomain = article.url.replace(/[^a-zA-Z0-9.-]/g, '').slice(0, 50);
      }
    }
    
    return crypto.createHash('md5')
      .update(`${titleWords}:${urlDomain}`)
      .digest('hex');
  }

  /**
   * Generate unique article ID
   */
  generateArticleId(title, url, source) {
    const content = `${title}:${url}:${source}:${Date.now()}`;
    return crypto.createHash('sha256').update(content).digest('hex').slice(0, 16);
  }

  /**
   * Cache ingestion results
   */
  async cacheResults(articles) {
    try {
      const cacheFile = path.join(this.cacheDir, `articles_${Date.now()}.json`);
      await fs.writeFile(cacheFile, JSON.stringify({
        timestamp: new Date().toISOString(),
        count: articles.length,
        articles
      }, null, 2));
      
      console.log(`💾 Cached ${articles.length} articles to ${cacheFile}`);
    } catch (error) {
      console.error('Failed to cache results:', error);
    }
  }

  // Utility methods
  cleanText(text) {
    if (!text) return '';
    return text.replace(/<[^>]*>/g, '').replace(/\s+/g, ' ').trim();
  }

  normalizeDate(dateStr) {
    try {
      return new Date(dateStr).toISOString();
    } catch {
      return new Date().toISOString();
    }
  }

  extractImageUrl(item) {
    if (item.enclosure?.['@_url']) return item.enclosure['@_url'];
    if (item['media:content']?.['@_url']) return item['media:content']['@_url'];
    if (item.image?.url) return item.image.url;
    return '';
  }

  extractTags(item) {
    const tags = [];
    if (item.category) {
      const cats = Array.isArray(item.category) ? item.category : [item.category];
      tags.push(...cats.map(c => typeof c === 'string' ? c : c['#text'] || c._ || '').filter(Boolean));
    }
    return tags;
  }

  chunkArray(array, size) {
    const chunks = [];
    for (let i = 0; i < array.length; i += size) {
      chunks.push(array.slice(i, i + size));
    }
    return chunks;
  }

  delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}

// Export for use in API routes and scheduled jobs
module.exports = {
  NewsIngestionEngine,
  NEWS_SOURCES,
  TOPIC_CATEGORIES
};

// CLI usage
if (require.main === module) {
  const engine = new NewsIngestionEngine();
  
  engine.ingestAllSources({
    query: process.argv[2],
    category: process.argv[3]
  }).then(results => {
    console.log('\n📊 Ingestion Results:');
    console.log(`Total articles: ${results.stats.totalFetched}`);
    console.log(`Duplicates removed: ${results.stats.duplicatesRemoved}`);
    console.log(`Processing time: ${results.stats.processingTime}ms`);
    console.log(`Errors: ${results.errors.length}`);
    
    if (results.errors.length > 0) {
      console.log('\n❌ Errors:');
      results.errors.forEach(err => console.log(`  - ${err.source}: ${err.error}`));
    }
  }).catch(console.error);
}