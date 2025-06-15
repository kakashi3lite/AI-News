/**
 * Batch Summarization API Route
 * Dr. NewsForge's AI News Dashboard
 * 
 * Handles bulk summarization of multiple articles with intelligent batching,
 * rate limiting, and comprehensive error handling.
 */

import { NLPSummarizationEngine } from '../../../../news/summarizer';

// Initialize the summarization engine with batch-optimized settings
const batchSummarizer = new NLPSummarizationEngine({
  cacheEnabled: true,
  maxConcurrent: 5, // Higher concurrency for batch processing
  cacheTTL: 24 * 60 * 60 * 1000 // 24 hours
});

// Rate limiting configuration
const RATE_LIMITS = {
  maxArticlesPerBatch: 50,
  maxBatchesPerHour: 20,
  maxContentLengthPerArticle: 5000
};

// Simple in-memory rate limiting (in production, use Redis)
const rateLimitStore = new Map();

export async function POST(request) {
  const startTime = Date.now();
  
  try {
    // Parse request body
    const body = await request.json();
    const { articles, options = {} } = body;
    
    // Validate request
    const validation = validateBatchRequest(articles, options);
    if (!validation.valid) {
      return Response.json({
        error: 'Validation Error',
        message: validation.message,
        code: validation.code
      }, { status: 400 });
    }
    
    // Check rate limits
    const clientId = getClientId(request);
    const rateLimitCheck = checkRateLimit(clientId);
    if (!rateLimitCheck.allowed) {
      return Response.json({
        error: 'Rate Limit Exceeded',
        message: rateLimitCheck.message,
        code: 'RATE_LIMIT'
      }, { 
        status: 429,
        headers: {
          'X-RateLimit-Limit': RATE_LIMITS.maxBatchesPerHour.toString(),
          'X-RateLimit-Remaining': rateLimitCheck.remaining.toString(),
          'X-RateLimit-Reset': rateLimitCheck.resetTime.toString()
        }
      });
    }
    
    console.log(`📚 Starting batch summarization for ${articles.length} articles`);
    
    // Configure batch options
    const batchOptions = {
      type: options.type || 'standard',
      style: options.style || 'abstractive',
      model: options.model || 'o4-mini-high',
      language: options.language || 'en',
      includeMetrics: options.includeMetrics || false
    };
    
    // Process batch
    const result = await batchSummarizer.summarizeBatch(articles, batchOptions);
    
    // Add request metadata
    const response = {
      ...result,
      requestId: generateRequestId(),
      processingTime: Date.now() - startTime,
      batchOptions,
      timestamp: new Date().toISOString()
    };
    
    // Update rate limit
    updateRateLimit(clientId);
    
    console.log(`✅ Batch summarization completed: ${result.stats.successful}/${result.stats.total} successful in ${response.processingTime}ms`);
    
    return Response.json(response, {
      headers: {
        'X-Processing-Time': response.processingTime.toString(),
        'X-Cache-Hit-Rate': (result.stats.cached / result.stats.total).toFixed(2)
      }
    });
    
  } catch (error) {
    console.error('❌ Batch summarization error:', error);
    
    // Handle specific error types
    if (error.message.includes('model error')) {
      return Response.json({
        error: 'Model Error',
        message: 'AI model is currently unavailable',
        code: 'MODEL_UNAVAILABLE'
      }, { status: 503 });
    }
    
    if (error.message.includes('timeout')) {
      return Response.json({
        error: 'Timeout Error',
        message: 'Batch processing timed out',
        code: 'PROCESSING_TIMEOUT'
      }, { status: 504 });
    }
    
    return Response.json({
      error: 'Internal Server Error',
      message: 'An unexpected error occurred during batch processing',
      code: 'INTERNAL_ERROR',
      details: process.env.NODE_ENV === 'development' ? error.message : undefined
    }, { status: 500 });
  }
}

// GET endpoint for batch status and configuration
export async function GET(request) {
  try {
    const url = new URL(request.url);
    const endpoint = url.searchParams.get('endpoint');
    
    if (endpoint === 'limits') {
      return Response.json({
        rateLimits: RATE_LIMITS,
        currentLimits: {
          maxArticlesPerBatch: RATE_LIMITS.maxArticlesPerBatch,
          maxBatchesPerHour: RATE_LIMITS.maxBatchesPerHour,
          maxContentLengthPerArticle: RATE_LIMITS.maxContentLengthPerArticle
        }
      });
    }
    
    if (endpoint === 'status') {
      const clientId = getClientId(request);
      const rateLimitInfo = getRateLimitInfo(clientId);
      
      return Response.json({
        status: 'operational',
        batchProcessor: {
          maxConcurrent: batchSummarizer.maxConcurrent,
          cacheEnabled: batchSummarizer.cacheEnabled,
          cacheSize: batchSummarizer.summaryCache.size
        },
        rateLimits: rateLimitInfo,
        timestamp: new Date().toISOString()
      });
    }
    
    return Response.json({
      error: 'Not Found',
      message: 'Invalid endpoint',
      code: 'INVALID_ENDPOINT'
    }, { status: 404 });
    
  } catch (error) {
    console.error('❌ Batch GET endpoint error:', error);
    return Response.json({
      error: 'Internal Server Error',
      message: 'Failed to process request',
      code: 'INTERNAL_ERROR'
    }, { status: 500 });
  }
}

/**
 * Validate batch summarization request
 */
function validateBatchRequest(articles, options) {
  if (!articles || !Array.isArray(articles)) {
    return {
      valid: false,
      message: 'Articles array is required',
      code: 'MISSING_ARTICLES'
    };
  }
  
  if (articles.length === 0) {
    return {
      valid: false,
      message: 'At least one article is required',
      code: 'EMPTY_ARTICLES'
    };
  }
  
  if (articles.length > RATE_LIMITS.maxArticlesPerBatch) {
    return {
      valid: false,
      message: `Maximum ${RATE_LIMITS.maxArticlesPerBatch} articles per batch`,
      code: 'BATCH_SIZE_EXCEEDED'
    };
  }
  
  // Validate each article
  for (let i = 0; i < articles.length; i++) {
    const article = articles[i];
    
    if (!article.id) {
      return {
        valid: false,
        message: `Article at index ${i} missing required 'id' field`,
        code: 'MISSING_ARTICLE_ID'
      };
    }
    
    const content = article.content || article.description || '';
    if (!content.trim()) {
      return {
        valid: false,
        message: `Article '${article.id}' has no content to summarize`,
        code: 'EMPTY_ARTICLE_CONTENT'
      };
    }
    
    if (content.length > RATE_LIMITS.maxContentLengthPerArticle) {
      return {
        valid: false,
        message: `Article '${article.id}' content exceeds ${RATE_LIMITS.maxContentLengthPerArticle} characters`,
        code: 'ARTICLE_TOO_LONG'
      };
    }
  }
  
  // Validate options
  if (options.type && !['brief', 'standard', 'detailed', 'executive'].includes(options.type)) {
    return {
      valid: false,
      message: 'Invalid summary type',
      code: 'INVALID_TYPE'
    };
  }
  
  if (options.style && !['extractive', 'abstractive', 'thematic', 'sentiment'].includes(options.style)) {
    return {
      valid: false,
      message: 'Invalid summary style',
      code: 'INVALID_STYLE'
    };
  }
  
  if (options.model && !['o4-mini-high', 'gpt-3.5-turbo', 'gpt-4'].includes(options.model)) {
    return {
      valid: false,
      message: 'Invalid model',
      code: 'INVALID_MODEL'
    };
  }
  
  return { valid: true };
}

/**
 * Get client identifier for rate limiting
 */
function getClientId(request) {
  // In production, use proper client identification
  const forwarded = request.headers.get('x-forwarded-for');
  const ip = forwarded ? forwarded.split(',')[0] : request.headers.get('x-real-ip') || 'unknown';
  const userAgent = request.headers.get('user-agent') || 'unknown';
  
  return `${ip}-${userAgent.substring(0, 50)}`;
}

/**
 * Check rate limits for client
 */
function checkRateLimit(clientId) {
  const now = Date.now();
  const hourWindow = 60 * 60 * 1000; // 1 hour
  const windowStart = now - hourWindow;
  
  if (!rateLimitStore.has(clientId)) {
    rateLimitStore.set(clientId, []);
  }
  
  const requests = rateLimitStore.get(clientId);
  
  // Remove old requests outside the window
  const recentRequests = requests.filter(timestamp => timestamp > windowStart);
  rateLimitStore.set(clientId, recentRequests);
  
  if (recentRequests.length >= RATE_LIMITS.maxBatchesPerHour) {
    return {
      allowed: false,
      message: `Maximum ${RATE_LIMITS.maxBatchesPerHour} batch requests per hour exceeded`,
      remaining: 0,
      resetTime: Math.ceil((recentRequests[0] + hourWindow) / 1000)
    };
  }
  
  return {
    allowed: true,
    remaining: RATE_LIMITS.maxBatchesPerHour - recentRequests.length,
    resetTime: Math.ceil((now + hourWindow) / 1000)
  };
}

/**
 * Update rate limit for client
 */
function updateRateLimit(clientId) {
  const now = Date.now();
  
  if (!rateLimitStore.has(clientId)) {
    rateLimitStore.set(clientId, []);
  }
  
  const requests = rateLimitStore.get(clientId);
  requests.push(now);
  rateLimitStore.set(clientId, requests);
}

/**
 * Get rate limit information for client
 */
function getRateLimitInfo(clientId) {
  const now = Date.now();
  const hourWindow = 60 * 60 * 1000;
  const windowStart = now - hourWindow;
  
  if (!rateLimitStore.has(clientId)) {
    return {
      requestsInWindow: 0,
      remaining: RATE_LIMITS.maxBatchesPerHour,
      resetTime: Math.ceil((now + hourWindow) / 1000)
    };
  }
  
  const requests = rateLimitStore.get(clientId);
  const recentRequests = requests.filter(timestamp => timestamp > windowStart);
  
  return {
    requestsInWindow: recentRequests.length,
    remaining: Math.max(0, RATE_LIMITS.maxBatchesPerHour - recentRequests.length),
    resetTime: recentRequests.length > 0 ? 
      Math.ceil((recentRequests[0] + hourWindow) / 1000) : 
      Math.ceil((now + hourWindow) / 1000)
  };
}

/**
 * Generate unique request ID
 */
function generateRequestId() {
  return `batch_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
}