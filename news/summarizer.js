/**
 * NLP Summarization Service
 * Dr. NewsForge's AI News Dashboard
 * 
 * Features:
 * - Transformer-based text summarization
 * - Multiple summarization strategies (extractive, abstractive)
 * - Batch processing capabilities
 * - Caching and performance optimization
 * - Integration with existing o4-mini-high and OpenAI models
 * - RESTful API endpoints
 * - Real-time processing with WebSocket support
 */

const axios = require('axios');
const crypto = require('crypto');
const fs = require('fs').promises;
const path = require('path');
const { performance } = require('perf_hooks');
const errorHandler = require('../lib/errorHandler');
const { ERROR_TYPES } = require('../lib/errorHandler');
const { queryO4Model } = require('../lib/o4ModelClient');
const { queryOpenAI } = require('../lib/openaiClient');

// Summarization configurations
const SUMMARIZATION_CONFIGS = {
  brief: {
    maxLength: 100,
    style: 'bullet-points',
    focus: 'key-facts'
  },
  standard: {
    maxLength: 250,
    style: 'paragraph',
    focus: 'comprehensive'
  },
  detailed: {
    maxLength: 500,
    style: 'structured',
    focus: 'analysis'
  },
  executive: {
    maxLength: 150,
    style: 'executive-summary',
    focus: 'business-impact'
  }
};

// Model configurations
const MODEL_CONFIGS = {
  'o4-mini-high': {
    provider: 'o4',
    maxTokens: 4000,
    temperature: 0.3,
    capabilities: ['summarization', 'analysis', 'extraction']
  },
  'gpt-3.5-turbo': {
    provider: 'openai',
    maxTokens: 4000,
    temperature: 0.3,
    capabilities: ['summarization', 'analysis', 'creative']
  },
  'gpt-4': {
    provider: 'openai',
    maxTokens: 8000,
    temperature: 0.2,
    capabilities: ['advanced-analysis', 'reasoning', 'summarization']
  }
};

// Prompt templates for different summarization types
const PROMPT_TEMPLATES = {
  extractive: {
    brief: "Extract the 3 most important facts from this article in bullet points:\n\n{content}\n\nKey Facts:",
    standard: "Summarize this article in 2-3 sentences focusing on the main points:\n\n{content}\n\nSummary:",
    detailed: "Provide a comprehensive summary of this article including context, main points, and implications:\n\n{content}\n\nDetailed Summary:"
  },
  abstractive: {
    brief: "Create a concise summary of this article in your own words (max 100 words):\n\n{content}\n\nSummary:",
    standard: "Write a clear, informative summary of this article (max 250 words):\n\n{content}\n\nSummary:",
    detailed: "Provide an in-depth analysis and summary of this article, including key insights and implications (max 500 words):\n\n{content}\n\nAnalysis:"
  },
  thematic: {
    brief: "Identify the main theme and 3 key points from this article:\n\n{content}\n\nTheme and Key Points:",
    standard: "Analyze the main themes and provide a structured summary:\n\n{content}\n\nThematic Summary:",
    detailed: "Provide a thematic analysis including main topics, subtopics, and their relationships:\n\n{content}\n\nThematic Analysis:"
  },
  sentiment: {
    brief: "Summarize this article and identify its overall sentiment (positive/negative/neutral):\n\n{content}\n\nSummary with Sentiment:",
    standard: "Provide a summary and detailed sentiment analysis of this article:\n\n{content}\n\nSentiment Analysis:",
    detailed: "Analyze the sentiment, tone, and emotional context of this article with a comprehensive summary:\n\n{content}\n\nDetailed Sentiment Analysis:"
  }
};

class NLPSummarizationEngine {
  constructor(options = {}) {
    this.cacheDir = options.cacheDir || path.join(__dirname, '../cache/summaries');
    this.defaultModel = options.defaultModel || 'o4-mini-high';
    this.maxConcurrent = options.maxConcurrent || 3;
    this.cacheEnabled = options.cacheEnabled !== false;
    this.cacheTTL = options.cacheTTL || 24 * 60 * 60 * 1000; // 24 hours
    
    this.summaryCache = new Map();
    this.processingQueue = [];
    this.isProcessing = false;
    
    this.initializeCache();
  }

  async initializeCache() {
    return await errorHandler.executeWithErrorHandling(
      async () => {
        await fs.mkdir(this.cacheDir, { recursive: true });
        console.log(`📁 Cache directory initialized: ${this.cacheDir}`);
        this.cacheEnabled = true;
      },
      {
        name: 'initialize_cache',
        retryOptions: {
          maxRetries: 2,
          retryableErrors: [ERROR_TYPES.FILESYSTEM]
        },
        fallback: async () => {
          console.log('⚠️  Cache initialization failed, disabling cache');
          this.cacheEnabled = false;
        }
      }
    );
  }

  /**
   * Main summarization method
   */
  async summarizeContent(content, options = {}) {
    return await errorHandler.executeWithErrorHandling(
      async () => {
        const config = {
          type: options.type || 'standard',
          style: options.style || 'abstractive',
          model: options.model || this.defaultModel,
          language: options.language || 'en',
          includeMetrics: options.includeMetrics || false,
          ...options
        };

        console.log(`🤖 Summarizing content with ${config.model} (${config.style}/${config.type})`);
        
        const startTime = Date.now();
        
        // Check cache first
        const cacheKey = this.generateCacheKey(content, config);
        if (this.cacheEnabled) {
          const cached = await this.getCachedSummary(cacheKey);
          if (cached) {
            console.log('📋 Using cached summary');
            return {
              ...cached,
              cached: true,
              processingTime: Date.now() - startTime
            };
          }
        }

        // Validate and prepare content
        const processedContent = this.preprocessContent(content);
        if (!processedContent) {
          throw new Error('Content is empty or invalid');
        }

        // Generate summary
        const summary = await this.generateSummary(processedContent, config);
        
        // Post-process and enhance
        const enhancedSummary = await this.enhanceSummary(summary, config);
        
        // Cache result
        if (this.cacheEnabled) {
          await this.cacheSummary(cacheKey, enhancedSummary);
        }

        const result = {
          ...enhancedSummary,
          cached: false,
          processingTime: Date.now() - startTime,
          config
        };

        console.log(`✅ Summary generated in ${result.processingTime}ms`);
        return result;
      },
      {
        name: 'summarize_content',
        retryOptions: {
          maxRetries: 2,
          retryableErrors: [ERROR_TYPES.NETWORK, ERROR_TYPES.EXTERNAL_API, ERROR_TYPES.RATE_LIMIT]
        },
        circuitBreakerOptions: {
          failureThreshold: 3,
          resetTimeout: 600000
        },
        fallback: async () => {
          console.log('🔄 Using fallback summarization...');
          return await this.fallbackSummarization(content, options);
        }
      }
    );
  }

  /**
   * Batch summarization for multiple articles
   */
  async summarizeBatch(articles, options = {}) {
    console.log(`📚 Starting batch summarization for ${articles.length} articles`);
    
    const results = {
      summaries: [],
      errors: [],
      stats: {
        total: articles.length,
        successful: 0,
        failed: 0,
        cached: 0,
        totalProcessingTime: 0
      }
    };

    const startTime = Date.now();
    
    // Process in chunks to respect rate limits
    const chunks = this.chunkArray(articles, this.maxConcurrent);
    
    for (const chunk of chunks) {
      const promises = chunk.map(async (article, index) => {
        try {
          const content = this.extractArticleContent(article);
          const summary = await this.summarizeContent(content, {
            ...options,
            articleId: article.id,
            title: article.title
          });
          
          results.summaries.push({
            articleId: article.id,
            title: article.title,
            ...summary
          });
          
          results.stats.successful++;
          if (summary.cached) results.stats.cached++;
          
        } catch (error) {
          console.error(`❌ Failed to summarize article ${article.id}:`, error.message);
          results.errors.push({
            articleId: article.id,
            title: article.title,
            error: error.message
          });
          results.stats.failed++;
        }
      });
      
      await Promise.all(promises);
      
      // Rate limiting between chunks
      if (chunks.indexOf(chunk) < chunks.length - 1) {
        await this.delay(1000);
      }
    }

    results.stats.totalProcessingTime = Date.now() - startTime;
    
    console.log(`✅ Batch summarization complete: ${results.stats.successful}/${results.stats.total} successful`);
    return results;
  }

  /**
   * Generate summary using specified model and configuration
   */
  async generateSummary(content, config) {
    const modelConfig = MODEL_CONFIGS[config.model];
    if (!modelConfig) {
      throw new Error(`Unsupported model: ${config.model}`);
    }

    // Get appropriate prompt template
    const promptTemplate = PROMPT_TEMPLATES[config.style]?.[config.type] || 
                          PROMPT_TEMPLATES.abstractive[config.type];
    
    if (!promptTemplate) {
      throw new Error(`No prompt template found for ${config.style}/${config.type}`);
    }

    // Build prompt
    const prompt = this.buildPrompt(promptTemplate, content, config);
    
    // Call appropriate model
    let response;
    if (modelConfig.provider === 'o4') {
      response = await this.callO4Model(prompt, config);
    } else if (modelConfig.provider === 'openai') {
      response = await this.callOpenAIModel(prompt, config);
    } else {
      throw new Error(`Unsupported model provider: ${modelConfig.provider}`);
    }

    return this.parseSummaryResponse(response, config);
  }

  /**
   * Call O4 model for summarization
   */
  async callO4Model(prompt, config) {
    try {
      const response = await queryO4Model(prompt);
      return response;
    } catch (error) {
      console.error('O4 model call failed:', error);
      throw new Error(`O4 model error: ${error.message}`);
    }
  }

  /**
   * Call OpenAI model for summarization
   */
  async callOpenAIModel(prompt, config) {
    try {
      const response = await queryOpenAI(prompt);
      return response;
    } catch (error) {
      console.error('OpenAI model call failed:', error);
      throw new Error(`OpenAI model error: ${error.message}`);
    }
  }

  /**
   * Build prompt from template and content
   */
  buildPrompt(template, content, config) {
    let prompt = template.replace('{content}', content);
    
    // Add additional context based on config
    if (config.title) {
      prompt = `Article Title: ${config.title}\n\n${prompt}`;
    }
    
    if (config.category) {
      prompt = `Category: ${config.category}\n${prompt}`;
    }
    
    // Add length constraints
    const summaryConfig = SUMMARIZATION_CONFIGS[config.type];
    if (summaryConfig) {
      prompt += `\n\nPlease limit the summary to approximately ${summaryConfig.maxLength} words.`;
    }
    
    return prompt;
  }

  /**
   * Parse and structure the model response
   */
  parseSummaryResponse(response, config) {
    const summary = {
      text: response.trim(),
      type: config.type,
      style: config.style,
      model: config.model,
      wordCount: this.countWords(response),
      generatedAt: new Date().toISOString()
    };

    // Extract structured data based on style
    if (config.style === 'extractive' && config.type === 'brief') {
      summary.keyPoints = this.extractBulletPoints(response);
    }
    
    if (config.style === 'sentiment') {
      summary.sentiment = this.extractSentiment(response);
    }
    
    if (config.style === 'thematic') {
      summary.themes = this.extractThemes(response);
    }

    return summary;
  }

  /**
   * Enhance summary with additional analysis
   */
  async enhanceSummary(summary, config) {
    const enhanced = { ...summary };
    
    // Add readability metrics
    enhanced.readability = this.calculateReadability(summary.text);
    
    // Add keyword extraction
    enhanced.keywords = this.extractKeywords(summary.text);
    
    // Add quality score
    enhanced.qualityScore = this.calculateQualityScore(summary, config);
    
    return enhanced;
  }

  /**
   * Extract article content for summarization
   */
  extractArticleContent(article) {
    // Prioritize content sources
    const content = article.content || article.description || article.title || '';
    
    if (!content.trim()) {
      throw new Error('Article has no content to summarize');
    }
    
    return content;
  }

  /**
   * Preprocess content before summarization
   */
  preprocessContent(content) {
    if (!content || typeof content !== 'string') {
      return null;
    }
    
    // Clean HTML tags
    let cleaned = content.replace(/<[^>]*>/g, '');
    
    // Normalize whitespace
    cleaned = cleaned.replace(/\s+/g, ' ').trim();
    
    // Remove very short content
    if (cleaned.length < 50) {
      return null;
    }
    
    // Truncate very long content
    if (cleaned.length > 10000) {
      cleaned = cleaned.substring(0, 10000) + '...';
    }
    
    return cleaned;
  }

  /**
   * Extract bullet points from response
   */
  extractBulletPoints(text) {
    const lines = text.split('\n');
    const points = [];
    
    for (const line of lines) {
      const trimmed = line.trim();
      if (trimmed.match(/^[•\-\*]\s+/) || trimmed.match(/^\d+\.\s+/)) {
        points.push(trimmed.replace(/^[•\-\*\d\.\s]+/, ''));
      }
    }
    
    return points.length > 0 ? points : null;
  }

  /**
   * Extract sentiment from response
   */
  extractSentiment(text) {
    const lowerText = text.toLowerCase();
    
    // Simple sentiment detection
    const positiveWords = ['positive', 'optimistic', 'good', 'success', 'growth', 'improvement'];
    const negativeWords = ['negative', 'pessimistic', 'bad', 'failure', 'decline', 'crisis'];
    
    const positiveCount = positiveWords.reduce((count, word) => 
      count + (lowerText.includes(word) ? 1 : 0), 0);
    const negativeCount = negativeWords.reduce((count, word) => 
      count + (lowerText.includes(word) ? 1 : 0), 0);
    
    if (positiveCount > negativeCount) return 'positive';
    if (negativeCount > positiveCount) return 'negative';
    return 'neutral';
  }

  /**
   * Extract themes from response
   */
  extractThemes(text) {
    // Simple theme extraction based on common patterns
    const themes = [];
    const lines = text.split('\n');
    
    for (const line of lines) {
      if (line.includes('Theme:') || line.includes('Topic:')) {
        const theme = line.split(':')[1]?.trim();
        if (theme) themes.push(theme);
      }
    }
    
    return themes.length > 0 ? themes : null;
  }

  /**
   * Extract keywords from text
   */
  extractKeywords(text) {
    const words = text.toLowerCase()
      .replace(/[^a-z0-9\s]/g, '')
      .split(/\s+/)
      .filter(word => word.length > 3);
    
    const frequency = {};
    words.forEach(word => {
      frequency[word] = (frequency[word] || 0) + 1;
    });
    
    return Object.entries(frequency)
      .sort(([,a], [,b]) => b - a)
      .slice(0, 10)
      .map(([word]) => word);
  }

  /**
   * Calculate readability score
   */
  calculateReadability(text) {
    const sentences = text.split(/[.!?]+/).filter(s => s.trim().length > 0);
    const words = text.split(/\s+/).filter(w => w.length > 0);
    const syllables = words.reduce((count, word) => count + this.countSyllables(word), 0);
    
    if (sentences.length === 0 || words.length === 0) return 0;
    
    // Flesch Reading Ease Score
    const score = 206.835 - (1.015 * (words.length / sentences.length)) - (84.6 * (syllables / words.length));
    return Math.max(0, Math.min(100, Math.round(score)));
  }

  /**
   * Count syllables in a word (approximation)
   */
  countSyllables(word) {
    word = word.toLowerCase();
    if (word.length <= 3) return 1;
    
    const vowels = word.match(/[aeiouy]+/g);
    let count = vowels ? vowels.length : 1;
    
    if (word.endsWith('e')) count--;
    if (count === 0) count = 1;
    
    return count;
  }

  /**
   * Calculate quality score for summary
   */
  calculateQualityScore(summary, config) {
    let score = 50; // Base score
    
    // Length appropriateness
    const targetLength = SUMMARIZATION_CONFIGS[config.type]?.maxLength || 250;
    const lengthRatio = summary.wordCount / targetLength;
    if (lengthRatio >= 0.7 && lengthRatio <= 1.2) {
      score += 20;
    } else if (lengthRatio >= 0.5 && lengthRatio <= 1.5) {
      score += 10;
    }
    
    // Readability
    if (summary.readability >= 60) score += 15;
    else if (summary.readability >= 40) score += 10;
    
    // Keyword diversity
    if (summary.keywords && summary.keywords.length >= 5) score += 10;
    
    // Structure (for bullet points)
    if (summary.keyPoints && summary.keyPoints.length >= 3) score += 5;
    
    return Math.min(100, Math.max(0, score));
  }

  /**
   * Count words in text
   */
  countWords(text) {
    return text.trim().split(/\s+/).filter(word => word.length > 0).length;
  }

  /**
   * Generate cache key for content and config
   */
  generateCacheKey(content, config) {
    const keyData = {
      content: content.substring(0, 500), // First 500 chars
      model: config.model,
      type: config.type,
      style: config.style
    };
    
    return crypto.createHash('md5')
      .update(JSON.stringify(keyData))
      .digest('hex');
  }

  /**
   * Fallback summarization using simple extraction
   */
  async fallbackSummarization(content, options = {}) {
    try {
      console.log('🔄 Using fallback summarization method...');
      
      if (Array.isArray(content)) {
        // Handle array of articles
        return {
          summaries: content.map(article => this.createFallbackSummary(article)),
          metadata: {
            totalArticles: content.length,
            processedArticles: content.length,
            processingTime: 0,
            method: 'fallback'
          }
        };
      } else {
        // Handle single content
        return this.createFallbackSummary(content);
      }
    } catch (error) {
      console.error('❌ Fallback summarization failed:', error.message);
      return this.createEmergencySummary(content);
    }
  }

  /**
   * Create a basic fallback summary
   */
  createFallbackSummary(article) {
    try {
      const content = article.content || article.description || article.title || '';
      const sentences = content.split(/[.!?]+/).filter(s => s.trim().length > 10);
      
      // Take first 2-3 sentences as summary
      const summary = sentences.slice(0, 3).join('. ').trim();
      
      return {
        id: article.id || 'fallback_' + Date.now(),
        title: article.title || 'Untitled',
        summary: summary || 'Summary not available',
        originalLength: content.length,
        summaryLength: summary.length,
        compressionRatio: content.length > 0 ? summary.length / content.length : 0,
        method: 'fallback_extraction',
        confidence: 0.5,
        cached: false,
        processingTime: 0
      };
    } catch (error) {
      console.error('❌ Failed to create fallback summary:', error.message);
      return this.createEmergencySummary(article);
    }
  }

  /**
   * Create emergency summary when all else fails
   */
  createEmergencySummary(content) {
    return {
      id: 'emergency_' + Date.now(),
      title: typeof content === 'object' ? (content.title || 'Content') : 'Content',
      summary: 'Summary temporarily unavailable due to processing issues.',
      originalLength: 0,
      summaryLength: 0,
      compressionRatio: 0,
      method: 'emergency',
      confidence: 0,
      cached: false,
      processingTime: 0,
      error: 'Summarization service unavailable'
    };
  }

  /**
   * Get cached summary
   */
  async getCachedSummary(cacheKey) {
    try {
      const cacheFile = path.join(this.cacheDir, `${cacheKey}.json`);
      const stats = await fs.stat(cacheFile);
      
      // Check if cache is still valid
      if (Date.now() - stats.mtime.getTime() > this.cacheTTL) {
        await fs.unlink(cacheFile);
        return null;
      }
      
      const cached = JSON.parse(await fs.readFile(cacheFile, 'utf8'));
      return cached;
      
    } catch (error) {
      return null;
    }
  }

  /**
   * Cache summary result
   */
  async cacheSummary(cacheKey, summary) {
    try {
      const cacheFile = path.join(this.cacheDir, `${cacheKey}.json`);
      await fs.writeFile(cacheFile, JSON.stringify(summary, null, 2));
    } catch (error) {
      console.warn('Failed to cache summary:', error.message);
    }
  }

  // Utility methods
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

// Export for use in API routes
module.exports = {
  NLPSummarizationEngine,
  SUMMARIZATION_CONFIGS,
  MODEL_CONFIGS,
  PROMPT_TEMPLATES
};

// CLI usage
if (require.main === module) {
  const engine = new NLPSummarizationEngine();
  
  const testContent = process.argv[2] || `
    Artificial intelligence is revolutionizing the healthcare industry through advanced 
    diagnostic tools, personalized treatment plans, and predictive analytics. Machine 
    learning algorithms can now analyze medical images with greater accuracy than human 
    radiologists in some cases, while natural language processing helps extract insights 
    from vast amounts of medical literature. The integration of AI in healthcare promises 
    to improve patient outcomes, reduce costs, and accelerate medical research.
  `;
  
  const config = {
    type: process.argv[3] || 'standard',
    style: process.argv[4] || 'abstractive',
    model: process.argv[5] || 'o4-mini-high'
  };
  
  engine.summarizeContent(testContent, config)
    .then(result => {
      console.log('\n📄 Summarization Result:');
      console.log(`Text: ${result.text}`);
      console.log(`Word Count: ${result.wordCount}`);
      console.log(`Quality Score: ${result.qualityScore}`);
      console.log(`Readability: ${result.readability}`);
      console.log(`Processing Time: ${result.processingTime}ms`);
      
      if (result.keywords) {
        console.log(`Keywords: ${result.keywords.join(', ')}`);
      }
      
      if (result.keyPoints) {
        console.log('Key Points:');
        result.keyPoints.forEach((point, i) => console.log(`  ${i + 1}. ${point}`));
      }
    })
    .catch(console.error);
}