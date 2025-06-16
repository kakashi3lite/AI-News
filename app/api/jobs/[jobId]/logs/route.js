import { NextResponse } from 'next/server';

// GET /api/jobs/[jobId]/logs
export async function GET(request, { params }) {
  try {
    const { jobId } = params;
    const { searchParams } = new URL(request.url);
    const level = searchParams.get('level'); // info, warn, error, debug
    const limit = parseInt(searchParams.get('limit')) || 100;
    const offset = parseInt(searchParams.get('offset')) || 0;
    const since = searchParams.get('since'); // ISO timestamp
    const until = searchParams.get('until'); // ISO timestamp
    const search = searchParams.get('search'); // Search in log messages
    
    if (!jobId) {
      return NextResponse.json(
        { error: 'Job ID is required' },
        { status: 400 }
      );
    }
    
    // Mock job logs data
    const mockJobLogs = {
      'job_001': [
        {
          id: 'log_001_001',
          timestamp: '2024-01-15T10:30:00.000Z',
          level: 'info',
          message: 'Job created and queued',
          context: {
            jobId: 'job_001',
            userId: 'user_123',
            queue: 'default',
            priority: 'normal'
          },
          source: 'job-scheduler',
          duration: null
        },
        {
          id: 'log_001_002',
          timestamp: '2024-01-15T10:30:05.123Z',
          level: 'info',
          message: 'Job started processing',
          context: {
            jobId: 'job_001',
            workerId: 'worker_001',
            attempt: 1,
            maxAttempts: 3
          },
          source: 'job-worker',
          duration: null
        },
        {
          id: 'log_001_003',
          timestamp: '2024-01-15T10:30:10.456Z',
          level: 'debug',
          message: 'Fetching article content from URL',
          context: {
            jobId: 'job_001',
            url: 'https://example.com/ai-research-2024',
            method: 'GET',
            userAgent: 'NewsBot/1.0'
          },
          source: 'content-fetcher',
          duration: null
        },
        {
          id: 'log_001_004',
          timestamp: '2024-01-15T10:30:15.789Z',
          level: 'info',
          message: 'Article content fetched successfully',
          context: {
            jobId: 'job_001',
            contentLength: 15420,
            contentType: 'text/html',
            statusCode: 200
          },
          source: 'content-fetcher',
          duration: 5333
        },
        {
          id: 'log_001_005',
          timestamp: '2024-01-15T10:30:20.012Z',
          level: 'debug',
          message: 'Preprocessing article content',
          context: {
            jobId: 'job_001',
            originalLength: 15420,
            cleanedLength: 12890,
            removedElements: ['ads', 'navigation', 'footer']
          },
          source: 'content-processor',
          duration: null
        },
        {
          id: 'log_001_006',
          timestamp: '2024-01-15T10:30:25.345Z',
          level: 'info',
          message: 'Starting AI summarization',
          context: {
            jobId: 'job_001',
            model: 'gpt-4-turbo',
            maxTokens: 500,
            temperature: 0.3
          },
          source: 'ai-summarizer',
          duration: null
        },
        {
          id: 'log_001_007',
          timestamp: '2024-01-15T10:31:30.678Z',
          level: 'info',
          message: 'AI summarization completed',
          context: {
            jobId: 'job_001',
            inputTokens: 15420,
            outputTokens: 342,
            cost: 0.0234,
            confidence: 0.92
          },
          source: 'ai-summarizer',
          duration: 65333
        },
        {
          id: 'log_001_008',
          timestamp: '2024-01-15T10:31:35.901Z',
          level: 'debug',
          message: 'Extracting key points from summary',
          context: {
            jobId: 'job_001',
            summaryLength: 342,
            keyPointsFound: 4
          },
          source: 'content-analyzer',
          duration: null
        },
        {
          id: 'log_001_009',
          timestamp: '2024-01-15T10:32:00.234Z',
          level: 'info',
          message: 'Performing sentiment analysis',
          context: {
            jobId: 'job_001',
            textLength: 342,
            model: 'sentiment-v2'
          },
          source: 'sentiment-analyzer',
          duration: null
        },
        {
          id: 'log_001_010',
          timestamp: '2024-01-15T10:32:10.567Z',
          level: 'info',
          message: 'Sentiment analysis completed',
          context: {
            jobId: 'job_001',
            sentiment: 'positive',
            confidence: 0.87,
            scores: {
              positive: 0.75,
              neutral: 0.20,
              negative: 0.05
            }
          },
          source: 'sentiment-analyzer',
          duration: 10333
        },
        {
          id: 'log_001_011',
          timestamp: '2024-01-15T10:32:15.890Z',
          level: 'info',
          message: 'Job completed successfully',
          context: {
            jobId: 'job_001',
            totalDuration: 130000,
            result: {
              summaryLength: 342,
              keyPoints: 4,
              sentiment: 'positive',
              readingTime: 8
            }
          },
          source: 'job-worker',
          duration: 130000
        }
      ],
      'job_002': [
        {
          id: 'log_002_001',
          timestamp: '2024-01-15T11:00:00.000Z',
          level: 'info',
          message: 'Job created and queued with high priority',
          context: {
            jobId: 'job_002',
            userId: 'system',
            queue: 'priority',
            priority: 'high'
          },
          source: 'job-scheduler',
          duration: null
        },
        {
          id: 'log_002_002',
          timestamp: '2024-01-15T11:00:02.123Z',
          level: 'info',
          message: 'Started scraping TechCrunch',
          context: {
            jobId: 'job_002',
            source: 'techcrunch',
            baseUrl: 'https://techcrunch.com',
            categories: ['ai', 'technology', 'science']
          },
          source: 'news-scraper',
          duration: null
        },
        {
          id: 'log_002_003',
          timestamp: '2024-01-15T11:00:15.456Z',
          level: 'debug',
          message: 'Fetching article list from category page',
          context: {
            jobId: 'job_002',
            url: 'https://techcrunch.com/category/artificial-intelligence/',
            expectedArticles: 20
          },
          source: 'news-scraper',
          duration: null
        },
        {
          id: 'log_002_004',
          timestamp: '2024-01-15T11:00:45.789Z',
          level: 'info',
          message: 'Found articles in AI category',
          context: {
            jobId: 'job_002',
            articlesFound: 18,
            duplicatesFiltered: 3,
            newArticles: 15
          },
          source: 'news-scraper',
          duration: 30333
        },
        {
          id: 'log_002_005',
          timestamp: '2024-01-15T11:01:30.012Z',
          level: 'info',
          message: 'TechCrunch scraping completed',
          context: {
            jobId: 'job_002',
            source: 'techcrunch',
            totalArticles: 35,
            processingTime: 87890,
            successRate: 0.97
          },
          source: 'news-scraper',
          duration: 87890
        },
        {
          id: 'log_002_006',
          timestamp: '2024-01-15T11:02:00.345Z',
          level: 'info',
          message: 'Started scraping Wired',
          context: {
            jobId: 'job_002',
            source: 'wired',
            baseUrl: 'https://wired.com',
            categories: ['ai', 'technology', 'science']
          },
          source: 'news-scraper',
          duration: null
        },
        {
          id: 'log_002_007',
          timestamp: '2024-01-15T11:02:30.678Z',
          level: 'warn',
          message: 'Rate limiting detected, slowing down requests',
          context: {
            jobId: 'job_002',
            source: 'wired',
            rateLimitHeaders: {
              'x-ratelimit-remaining': '5',
              'x-ratelimit-reset': '1642248180'
            },
            delayAdded: 2000
          },
          source: 'news-scraper',
          duration: null
        },
        {
          id: 'log_002_008',
          timestamp: '2024-01-15T11:03:15.901Z',
          level: 'info',
          message: 'Wired scraping completed',
          context: {
            jobId: 'job_002',
            source: 'wired',
            totalArticles: 28,
            processingTime: 75556,
            rateLimitEncountered: true
          },
          source: 'news-scraper',
          duration: 75556
        },
        {
          id: 'log_002_009',
          timestamp: '2024-01-15T11:03:20.234Z',
          level: 'info',
          message: 'Started scraping Ars Technica',
          context: {
            jobId: 'job_002',
            source: 'ars-technica',
            baseUrl: 'https://arstechnica.com',
            categories: ['ai', 'technology', 'science']
          },
          source: 'news-scraper',
          duration: null
        }
      ],
      'job_003': [
        {
          id: 'log_003_001',
          timestamp: '2024-01-15T11:15:00.000Z',
          level: 'info',
          message: 'Job created and queued',
          context: {
            jobId: 'job_003',
            userId: 'user_456',
            queue: 'default',
            priority: 'normal'
          },
          source: 'job-scheduler',
          duration: null
        },
        {
          id: 'log_003_002',
          timestamp: '2024-01-15T11:15:05.123Z',
          level: 'info',
          message: 'Started sentiment analysis',
          context: {
            jobId: 'job_003',
            articleId: 'article_789',
            commentCount: 3,
            analysisType: 'detailed'
          },
          source: 'sentiment-analyzer',
          duration: null
        },
        {
          id: 'log_003_003',
          timestamp: '2024-01-15T11:15:08.456Z',
          level: 'debug',
          message: 'Fetching comment data',
          context: {
            jobId: 'job_003',
            commentIds: ['comment_1', 'comment_2', 'comment_3'],
            batchSize: 3
          },
          source: 'data-fetcher',
          duration: null
        },
        {
          id: 'log_003_004',
          timestamp: '2024-01-15T11:15:10.789Z',
          level: 'error',
          message: 'API rate limit exceeded on first attempt',
          context: {
            jobId: 'job_003',
            apiEndpoint: 'https://api.sentiment.com/analyze',
            statusCode: 429,
            rateLimitReset: '2024-01-15T11:20:00Z',
            retryAfter: 300
          },
          source: 'sentiment-api',
          duration: 2333
        },
        {
          id: 'log_003_005',
          timestamp: '2024-01-15T11:16:00.012Z',
          level: 'info',
          message: 'Retrying sentiment analysis (attempt 2)',
          context: {
            jobId: 'job_003',
            attempt: 2,
            maxAttempts: 3,
            waitTime: 50000
          },
          source: 'job-worker',
          duration: null
        },
        {
          id: 'log_003_006',
          timestamp: '2024-01-15T11:16:30.345Z',
          level: 'error',
          message: 'API rate limit exceeded on second attempt',
          context: {
            jobId: 'job_003',
            apiEndpoint: 'https://api.sentiment.com/analyze',
            statusCode: 429,
            rateLimitReset: '2024-01-15T11:20:00Z',
            retryAfter: 210,
            consecutiveFailures: 2
          },
          source: 'sentiment-api',
          duration: 30333
        },
        {
          id: 'log_003_007',
          timestamp: '2024-01-15T11:16:35.678Z',
          level: 'warn',
          message: 'Job will retry after rate limit reset',
          context: {
            jobId: 'job_003',
            nextRetry: '2024-01-15T11:20:00Z',
            remainingAttempts: 1,
            backoffStrategy: 'exponential'
          },
          source: 'job-scheduler',
          duration: null
        }
      ]
    };
    
    let logs = mockJobLogs[jobId] || [];
    
    if (!logs.length) {
      return NextResponse.json(
        { error: 'Job not found or no logs available' },
        { status: 404 }
      );
    }
    
    // Apply filters
    if (level) {
      logs = logs.filter(log => log.level === level);
    }
    
    if (since) {
      const sinceDate = new Date(since);
      logs = logs.filter(log => new Date(log.timestamp) >= sinceDate);
    }
    
    if (until) {
      const untilDate = new Date(until);
      logs = logs.filter(log => new Date(log.timestamp) <= untilDate);
    }
    
    if (search) {
      const searchLower = search.toLowerCase();
      logs = logs.filter(log => 
        log.message.toLowerCase().includes(searchLower) ||
        log.source.toLowerCase().includes(searchLower) ||
        JSON.stringify(log.context).toLowerCase().includes(searchLower)
      );
    }
    
    // Apply pagination
    const totalLogs = logs.length;
    const paginatedLogs = logs.slice(offset, offset + limit);
    
    // Calculate statistics
    const stats = {
      total: totalLogs,
      byLevel: {
        info: logs.filter(l => l.level === 'info').length,
        warn: logs.filter(l => l.level === 'warn').length,
        error: logs.filter(l => l.level === 'error').length,
        debug: logs.filter(l => l.level === 'debug').length
      },
      bySources: logs.reduce((acc, log) => {
        acc[log.source] = (acc[log.source] || 0) + 1;
        return acc;
      }, {}),
      timeRange: logs.length > 0 ? {
        start: logs[0].timestamp,
        end: logs[logs.length - 1].timestamp
      } : null
    };
    
    return NextResponse.json({
      jobId,
      logs: paginatedLogs,
      pagination: {
        offset,
        limit,
        total: totalLogs,
        hasMore: offset + limit < totalLogs
      },
      stats,
      filters: {
        level,
        since,
        until,
        search
      }
    });
  } catch (error) {
    console.error(`[/api/jobs/${params?.jobId}/logs] Error:`, error);
    return NextResponse.json(
      { error: error.message },
      { status: 500 }
    );
  }
}