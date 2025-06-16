import { NextResponse } from 'next/server';

// GET /api/jobs
export async function GET(request) {
  try {
    const { searchParams } = new URL(request.url);
    const status = searchParams.get('status'); // pending, running, completed, failed
    const type = searchParams.get('type'); // summarization, analysis, scraping, etc.
    const priority = searchParams.get('priority'); // low, normal, high, urgent
    const page = parseInt(searchParams.get('page')) || 1;
    const limit = parseInt(searchParams.get('limit')) || 20;
    const userId = searchParams.get('userId');
    const queue = searchParams.get('queue'); // default, priority, background
    
    // Mock job data
    const mockJobs = [
      {
        id: 'job_001',
        type: 'article_summarization',
        status: 'completed',
        priority: 'normal',
        queue: 'default',
        userId: 'user_123',
        title: 'Summarize AI Research Article',
        description: 'Generate summary for "Advances in Neural Networks 2024"',
        payload: {
          articleId: 'article_456',
          url: 'https://example.com/ai-research-2024',
          summaryType: 'detailed'
        },
        result: {
          summary: 'This article discusses recent advances in neural network architectures...',
          keyPoints: ['Transformer improvements', 'Efficiency gains', 'New applications'],
          readingTime: 8
        },
        progress: 100,
        createdAt: '2024-01-15T10:30:00Z',
        startedAt: '2024-01-15T10:30:05Z',
        completedAt: '2024-01-15T10:32:15Z',
        duration: 130,
        attempts: 1,
        maxAttempts: 3,
        error: null
      },
      {
        id: 'job_002',
        type: 'news_scraping',
        status: 'running',
        priority: 'high',
        queue: 'priority',
        userId: 'system',
        title: 'Scrape Tech News Sources',
        description: 'Collect latest articles from configured news sources',
        payload: {
          sources: ['techcrunch', 'wired', 'ars-technica'],
          categories: ['ai', 'technology', 'science'],
          maxArticles: 100
        },
        result: null,
        progress: 65,
        createdAt: '2024-01-15T11:00:00Z',
        startedAt: '2024-01-15T11:00:02Z',
        completedAt: null,
        duration: null,
        attempts: 1,
        maxAttempts: 3,
        error: null,
        estimatedCompletion: '2024-01-15T11:05:00Z'
      },
      {
        id: 'job_003',
        type: 'sentiment_analysis',
        status: 'pending',
        priority: 'normal',
        queue: 'default',
        userId: 'user_456',
        title: 'Analyze Article Sentiment',
        description: 'Perform sentiment analysis on user comments',
        payload: {
          articleId: 'article_789',
          commentIds: ['comment_1', 'comment_2', 'comment_3']
        },
        result: null,
        progress: 0,
        createdAt: '2024-01-15T11:15:00Z',
        startedAt: null,
        completedAt: null,
        duration: null,
        attempts: 0,
        maxAttempts: 3,
        error: null,
        estimatedStart: '2024-01-15T11:20:00Z'
      },
      {
        id: 'job_004',
        type: 'data_export',
        status: 'failed',
        priority: 'low',
        queue: 'background',
        userId: 'user_789',
        title: 'Export User Data',
        description: 'Generate CSV export of user activity data',
        payload: {
          userId: 'user_789',
          dateRange: { start: '2024-01-01', end: '2024-01-15' },
          format: 'csv',
          includePersonalData: false
        },
        result: null,
        progress: 0,
        createdAt: '2024-01-15T09:45:00Z',
        startedAt: '2024-01-15T09:45:03Z',
        completedAt: null,
        duration: null,
        attempts: 3,
        maxAttempts: 3,
        error: {
          code: 'EXPORT_FAILED',
          message: 'Database connection timeout during export',
          timestamp: '2024-01-15T09:47:30Z'
        },
        nextRetry: null
      },
      {
        id: 'job_005',
        type: 'image_processing',
        status: 'completed',
        priority: 'normal',
        queue: 'default',
        userId: 'user_321',
        title: 'Process Article Images',
        description: 'Optimize and generate thumbnails for article images',
        payload: {
          articleId: 'article_101',
          imageUrls: [
            'https://example.com/image1.jpg',
            'https://example.com/image2.png'
          ],
          sizes: ['thumbnail', 'medium', 'large']
        },
        result: {
          processedImages: [
            {
              original: 'https://example.com/image1.jpg',
              thumbnail: 'https://cdn.example.com/thumb_image1.jpg',
              medium: 'https://cdn.example.com/med_image1.jpg',
              large: 'https://cdn.example.com/large_image1.jpg'
            }
          ],
          totalSize: '2.4MB',
          compressionRatio: 0.65
        },
        progress: 100,
        createdAt: '2024-01-15T08:30:00Z',
        startedAt: '2024-01-15T08:30:01Z',
        completedAt: '2024-01-15T08:32:45Z',
        duration: 164,
        attempts: 1,
        maxAttempts: 3,
        error: null
      }
    ];
    
    // Filter jobs based on query parameters
    let filteredJobs = mockJobs;
    
    if (status) {
      filteredJobs = filteredJobs.filter(job => job.status === status);
    }
    
    if (type) {
      filteredJobs = filteredJobs.filter(job => job.type === type);
    }
    
    if (priority) {
      filteredJobs = filteredJobs.filter(job => job.priority === priority);
    }
    
    if (userId) {
      filteredJobs = filteredJobs.filter(job => job.userId === userId);
    }
    
    if (queue) {
      filteredJobs = filteredJobs.filter(job => job.queue === queue);
    }
    
    // Pagination
    const startIndex = (page - 1) * limit;
    const endIndex = startIndex + limit;
    const paginatedJobs = filteredJobs.slice(startIndex, endIndex);
    
    // Calculate statistics
    const stats = {
      total: mockJobs.length,
      filtered: filteredJobs.length,
      byStatus: {
        pending: mockJobs.filter(j => j.status === 'pending').length,
        running: mockJobs.filter(j => j.status === 'running').length,
        completed: mockJobs.filter(j => j.status === 'completed').length,
        failed: mockJobs.filter(j => j.status === 'failed').length
      },
      byPriority: {
        low: mockJobs.filter(j => j.priority === 'low').length,
        normal: mockJobs.filter(j => j.priority === 'normal').length,
        high: mockJobs.filter(j => j.priority === 'high').length,
        urgent: mockJobs.filter(j => j.priority === 'urgent').length
      },
      byQueue: {
        default: mockJobs.filter(j => j.queue === 'default').length,
        priority: mockJobs.filter(j => j.queue === 'priority').length,
        background: mockJobs.filter(j => j.queue === 'background').length
      },
      avgDuration: Math.round(
        mockJobs
          .filter(j => j.duration)
          .reduce((sum, j) => sum + j.duration, 0) /
        mockJobs.filter(j => j.duration).length
      ),
      successRate: (
        mockJobs.filter(j => j.status === 'completed').length /
        mockJobs.filter(j => j.status !== 'pending' && j.status !== 'running').length
      ).toFixed(2)
    };
    
    return NextResponse.json({
      jobs: paginatedJobs,
      pagination: {
        page,
        limit,
        total: filteredJobs.length,
        pages: Math.ceil(filteredJobs.length / limit),
        hasNext: endIndex < filteredJobs.length,
        hasPrev: page > 1
      },
      stats,
      filters: {
        status,
        type,
        priority,
        userId,
        queue
      }
    });
  } catch (error) {
    console.error('[/api/jobs] Error:', error);
    return NextResponse.json(
      { error: error.message },
      { status: 500 }
    );
  }
}

// POST /api/jobs
export async function POST(request) {
  try {
    const body = await request.json();
    const {
      type,
      title,
      description,
      payload,
      priority = 'normal',
      queue = 'default',
      userId,
      scheduledFor,
      maxAttempts = 3,
      timeout = 300000 // 5 minutes default
    } = body;
    
    // Validate required fields
    if (!type || !title || !payload) {
      return NextResponse.json(
        { error: 'Missing required fields: type, title, payload' },
        { status: 400 }
      );
    }
    
    // Validate job type
    const validTypes = [
      'article_summarization',
      'news_scraping',
      'sentiment_analysis',
      'data_export',
      'image_processing',
      'email_notification',
      'data_backup',
      'cache_refresh',
      'analytics_report'
    ];
    
    if (!validTypes.includes(type)) {
      return NextResponse.json(
        { error: `Invalid job type. Must be one of: ${validTypes.join(', ')}` },
        { status: 400 }
      );
    }
    
    // Validate priority
    const validPriorities = ['low', 'normal', 'high', 'urgent'];
    if (!validPriorities.includes(priority)) {
      return NextResponse.json(
        { error: `Invalid priority. Must be one of: ${validPriorities.join(', ')}` },
        { status: 400 }
      );
    }
    
    // Validate queue
    const validQueues = ['default', 'priority', 'background'];
    if (!validQueues.includes(queue)) {
      return NextResponse.json(
        { error: `Invalid queue. Must be one of: ${validQueues.join(', ')}` },
        { status: 400 }
      );
    }
    
    // Create new job
    const newJob = {
      id: `job_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      type,
      title,
      description,
      payload,
      priority,
      queue,
      userId: userId || 'anonymous',
      status: scheduledFor ? 'scheduled' : 'pending',
      progress: 0,
      result: null,
      error: null,
      attempts: 0,
      maxAttempts,
      timeout,
      createdAt: new Date().toISOString(),
      scheduledFor: scheduledFor || null,
      startedAt: null,
      completedAt: null,
      duration: null
    };
    
    console.log('Created new job:', newJob);
    
    // In real implementation, add job to queue/database
    
    return NextResponse.json({
      success: true,
      job: newJob,
      message: 'Job created successfully',
      estimatedStart: scheduledFor || new Date(Date.now() + 30000).toISOString() // 30 seconds from now
    });
  } catch (error) {
    console.error('[/api/jobs] POST Error:', error);
    return NextResponse.json(
      { error: error.message },
      { status: 500 }
    );
  }
}