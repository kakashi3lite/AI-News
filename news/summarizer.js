/**
 * NLP Summarization Orchestrator
 * Coordinates content preprocessing, summary generation, quality analysis, and caching.
 */

const crypto = require('crypto');
const fs = require('fs').promises;
const path = require('path');
const errorHandler = require('../lib/errorHandler');
const { ERROR_TYPES } = require('../lib/errorHandler');
const { SUMMARIZATION_CONFIGS, MODEL_CONFIGS, PROMPT_TEMPLATES } = require('./summarizationConfig');
const { preprocessContent, extractArticleContent } = require('./contentPreprocessor');
const { generateSummary } = require('./summaryGenerator');
const { enhanceSummary } = require('./qualityAnalyzer');

const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

const chunkArray = (array, size) => {
  const chunks = [];
  for (let i = 0; i < array.length; i += size) chunks.push(array.slice(i, i + size));
  return chunks;
};

class NLPSummarizationEngine {
  constructor(options = {}) {
    this.cacheDir = options.cacheDir || path.join(__dirname, '../cache/summaries');
    this.defaultModel = options.defaultModel || 'o4-mini-high';
    this.maxConcurrent = options.maxConcurrent || 3;
    this.cacheEnabled = options.cacheEnabled !== false;
    this.cacheTTL = options.cacheTTL || 24 * 60 * 60 * 1000;
    this.initializeCache();
  }

  async initializeCache() {
    return await errorHandler.executeWithErrorHandling(
      async () => {
        await fs.mkdir(this.cacheDir, { recursive: true });
        this.cacheEnabled = true;
      },
      { name: 'initialize_cache', retryOptions: { maxRetries: 2 }, fallback: async () => { this.cacheEnabled = false; } }
    );
  }

  async summarizeContent(content, options = {}) {
    return await errorHandler.executeWithErrorHandling(
      async () => {
        const config = { type: options.type || 'standard', style: options.style || 'abstractive', model: options.model || this.defaultModel, language: options.language || 'en', ...options };
        const startTime = Date.now();

        const cacheKey = this.generateCacheKey(content, config);
        if (this.cacheEnabled) {
          const cached = await this.getCachedSummary(cacheKey);
          if (cached) return { ...cached, cached: true, processingTime: Date.now() - startTime };
        }

        const processedContent = preprocessContent(content);
        if (!processedContent) throw new Error('Content is empty or invalid');

        const summary = await generateSummary(processedContent, config);
        const enhanced = enhanceSummary(summary, config);

        if (this.cacheEnabled) await this.cacheSummary(cacheKey, enhanced);

        return { ...enhanced, cached: false, processingTime: Date.now() - startTime, config };
      },
      {
        name: 'summarize_content',
        retryOptions: { maxRetries: 2, retryableErrors: [ERROR_TYPES.NETWORK, ERROR_TYPES.EXTERNAL_API, ERROR_TYPES.RATE_LIMIT] },
        circuitBreakerOptions: { failureThreshold: 3, resetTimeout: 600000 },
        fallback: async () => this.fallbackSummarization(content, options)
      }
    );
  }

  async summarizeBatch(articles, options = {}) {
    console.log(`📚 Starting batch summarization for ${articles.length} articles`);
    const results = { summaries: [], errors: [], stats: { total: articles.length, successful: 0, failed: 0, cached: 0, totalProcessingTime: 0 } };
    const startTime = Date.now();

    for (const chunk of chunkArray(articles, this.maxConcurrent)) {
      await Promise.all(chunk.map(async (article) => {
        try {
          const content = extractArticleContent(article);
          const summary = await this.summarizeContent(content, { ...options, articleId: article.id, title: article.title });
          results.summaries.push({ articleId: article.id, title: article.title, ...summary });
          results.stats.successful++;
          if (summary.cached) results.stats.cached++;
        } catch (error) {
          results.errors.push({ articleId: article.id, title: article.title, error: error.message });
          results.stats.failed++;
        }
      }));
      await delay(1000);
    }

    results.stats.totalProcessingTime = Date.now() - startTime;
    console.log(`✅ Batch summarization complete: ${results.stats.successful}/${results.stats.total} successful`);
    return results;
  }

  async fallbackSummarization(content) {
    if (Array.isArray(content)) {
      return { summaries: content.map((a) => this.createFallbackSummary(a)), metadata: { totalArticles: content.length, method: 'fallback' } };
    }
    return this.createFallbackSummary(content);
  }

  createFallbackSummary(article) {
    const content = typeof article === 'string' ? article : (article.content || article.description || article.title || '');
    const sentences = content.split(/[.!?]+/).filter((s) => s.trim().length > 10);
    const summary = sentences.slice(0, 3).join('. ').trim();
    return {
      id: (typeof article === 'object' ? article.id : null) || 'fallback_' + Date.now(),
      title: typeof article === 'object' ? (article.title || 'Untitled') : 'Untitled',
      summary: summary || 'Summary not available',
      method: 'fallback_extraction',
      confidence: 0.5,
      cached: false,
      processingTime: 0
    };
  }

  generateCacheKey(content, config) {
    return crypto.createHash('md5').update(JSON.stringify({
      content: content.substring(0, 500),
      model: config.model,
      type: config.type,
      style: config.style
    })).digest('hex');
  }

  async getCachedSummary(cacheKey) {
    try {
      const cacheFile = path.join(this.cacheDir, `${cacheKey}.json`);
      const stats = await fs.stat(cacheFile);
      if (Date.now() - stats.mtime.getTime() > this.cacheTTL) {
        await fs.unlink(cacheFile);
        return null;
      }
      return JSON.parse(await fs.readFile(cacheFile, 'utf8'));
    } catch {
      return null;
    }
  }

  async cacheSummary(cacheKey, summary) {
    try {
      await fs.writeFile(path.join(this.cacheDir, `${cacheKey}.json`), JSON.stringify(summary, null, 2));
    } catch (error) {
      console.warn('Failed to cache summary:', error.message);
    }
  }
}

module.exports = { NLPSummarizationEngine, SUMMARIZATION_CONFIGS, MODEL_CONFIGS, PROMPT_TEMPLATES };

if (require.main === module) {
  const engine = new NLPSummarizationEngine();
  const testContent = process.argv[2] || 'Artificial intelligence is revolutionizing the healthcare industry through advanced diagnostic tools, personalized treatment plans, and predictive analytics.';
  engine.summarizeContent(testContent, {
    type: process.argv[3] || 'standard',
    style: process.argv[4] || 'abstractive',
    model: process.argv[5] || 'o4-mini-high'
  }).then((result) => {
    console.log(`\n📄 Summary: ${result.text}`);
    console.log(`Quality: ${result.qualityScore}, Readability: ${result.readability}, Time: ${result.processingTime}ms`);
  }).catch(console.error);
}
