import { NLPSummarizationEngine } from '../../../news/summarizer';

// Initialize the summarization engine
const summarizer = new NLPSummarizationEngine({
  cacheEnabled: true,
  maxConcurrent: 3,
  cacheTTL: 24 * 60 * 60 * 1000 // 24 hours
});

export async function POST(request) {
  try {
    const body = await request.json();
    const { content, type, style, model, includeMetrics, title, category } = body;
    
    if (!content) {
      return Response.json({ 
        error: 'Validation Error',
        message: 'Content is required',
        code: 'INVALID_INPUT'
      }, { status: 400 });
    }

    if (content.length < 50) {
      return Response.json({ 
        error: 'Validation Error',
        message: 'Content must be at least 50 characters long',
        code: 'CONTENT_TOO_SHORT'
      }, { status: 400 });
    }

    if (content.length > 10000) {
      return Response.json({ 
        error: 'Validation Error',
        message: 'Content must be less than 10,000 characters',
        code: 'CONTENT_TOO_LONG'
      }, { status: 400 });
    }

    // Configure summarization options
    const options = {
      type: type || 'standard',
      style: style || 'abstractive',
      model: model || 'o4-mini-high',
      includeMetrics: includeMetrics || false,
      title,
      category
    };

    console.log(`📝 Summarizing content with ${options.model} (${options.style}/${options.type})`);
    
    const result = await summarizer.summarizeContent(content, options);
    
    return Response.json(result);
    
  } catch (error) {
    console.error('❌ Summarization error:', error);
    
    // Handle specific error types
    if (error.message.includes('model error')) {
      return Response.json({
        error: 'Model Error',
        message: 'AI model is currently unavailable',
        code: 'MODEL_UNAVAILABLE'
      }, { status: 503 });
    }
    
    if (error.message.includes('rate limit')) {
      return Response.json({
        error: 'Rate Limit Exceeded',
        message: 'Too many requests, please try again later',
        code: 'RATE_LIMIT'
      }, { status: 429 });
    }
    
    return Response.json({
      error: 'Internal Server Error',
      message: 'An unexpected error occurred while processing your request',
      code: 'INTERNAL_ERROR',
      details: process.env.NODE_ENV === 'development' ? error.message : undefined
    }, { status: 500 });
  }
}

// GET endpoint for health check
export async function GET(request) {
  try {
    const url = new URL(request.url);
    const endpoint = url.searchParams.get('endpoint');
    
    if (endpoint === 'health') {
      // Health check
      const health = {
        status: 'healthy',
        timestamp: new Date().toISOString(),
        models: {
          'o4-mini-high': 'available',
          'gpt-3.5-turbo': 'available'
        },
        cache: {
          status: 'active',
          size: summarizer.summaryCache.size
        }
      };
      
      return Response.json(health);
    }
    
    if (endpoint === 'models') {
      // Available models
      const { MODEL_CONFIGS } = require('../../../news/summarizer');
      const models = Object.entries(MODEL_CONFIGS).map(([name, config]) => ({
        name,
        provider: config.provider,
        maxTokens: config.maxTokens,
        capabilities: config.capabilities,
        available: true
      }));
      
      return Response.json({ models });
    }
    
    return Response.json({
      error: 'Not Found',
      message: 'Invalid endpoint',
      code: 'INVALID_ENDPOINT'
    }, { status: 404 });
    
  } catch (error) {
    console.error('❌ GET endpoint error:', error);
    return Response.json({
      error: 'Internal Server Error',
      message: 'Failed to process request',
      code: 'INTERNAL_ERROR'
    }, { status: 500 });
  }
}
