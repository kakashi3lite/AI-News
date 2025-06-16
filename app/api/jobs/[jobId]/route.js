import { NextResponse } from 'next/server';

// GET /api/jobs/[jobId]
export async function GET(request, { params }) {
  try {
    const { jobId } = params;
    
    if (!jobId) {
      return NextResponse.json(
        { error: 'Job ID is required' },
        { status: 400 }
      );
    }
    
    // Mock job data - in real implementation, fetch from database
    const mockJobs = {
      'job_001': {
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
          summaryType: 'detailed',
          maxLength: 500
        },
        result: {
          summary: 'This article discusses recent advances in neural network architectures, focusing on transformer improvements and efficiency gains. Key developments include attention mechanism optimizations, reduced computational requirements, and new applications in natural language processing.',
          keyPoints: [
            'Transformer architecture improvements reduce computational overhead by 40%',
            'New attention mechanisms enable better long-range dependency modeling',
            'Applications expanded to multimodal tasks including vision-language models',
            'Efficiency gains make deployment feasible on edge devices'
          ],
          readingTime: 8,
          sentiment: 'positive',
          topics: ['artificial intelligence', 'neural networks', 'transformers', 'efficiency'],
          confidence: 0.92
        },
        progress: 100,
        createdAt: '2024-01-15T10:30:00Z',
        startedAt: '2024-01-15T10:30:05Z',
        completedAt: '2024-01-15T10:32:15Z',
        duration: 130,
        attempts: 1,
        maxAttempts: 3,
        timeout: 300000,
        error: null,
        logs: [
          {
            timestamp: '2024-01-15T10:30:00Z',
            level: 'info',
            message: 'Job created and queued'
          },
          {
            timestamp: '2024-01-15T10:30:05Z',
            level: 'info',
            message: 'Job started processing'
          },
          {
            timestamp: '2024-01-15T10:30:15Z',
            level: 'info',
            message: 'Article content fetched successfully'
          },
          {
            timestamp: '2024-01-15T10:31:30Z',
            level: 'info',
            message: 'AI summarization completed'
          },
          {
            timestamp: '2024-01-15T10:32:15Z',
            level: 'info',
            message: 'Job completed successfully'
          }
        ],
        metrics: {
          inputTokens: 15420,
          outputTokens: 342,
          processingTime: 125,
          apiCalls: 2,
          cacheHits: 1
        }
      },
      'job_002': {
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
          maxArticles: 100,
          dateRange: { hours: 24 }
        },
        result: null,
        progress: 65,
        createdAt: '2024-01-15T11:00:00Z',
        startedAt: '2024-01-15T11:00:02Z',
        completedAt: null,
        duration: null,
        attempts: 1,
        maxAttempts: 3,
        timeout: 600000,
        error: null,
        estimatedCompletion: '2024-01-15T11:05:00Z',
        logs: [
          {
            timestamp: '2024-01-15T11:00:00Z',
            level: 'info',
            message: 'Job created and queued with high priority'
          },
          {
            timestamp: '2024-01-15T11:00:02Z',
            level: 'info',
            message: 'Started scraping TechCrunch'
          },
          {
            timestamp: '2024-01-15T11:01:30Z',
            level: 'info',
            message: 'TechCrunch scraping completed - 35 articles found'
          },
          {
            timestamp: '2024-01-15T11:02:00Z',
            level: 'info',
            message: 'Started scraping Wired'
          },
          {
            timestamp: '2024-01-15T11:03:15Z',
            level: 'info',
            message: 'Wired scraping completed - 28 articles found'
          },
          {
            timestamp: '2024-01-15T11:03:20Z',
            level: 'info',
            message: 'Started scraping Ars Technica'
          }
        ],
        metrics: {
          articlesFound: 63,
          articlesProcessed: 63,
          duplicatesRemoved: 8,
          sourcesCompleted: 2,
          sourcesRemaining: 1,
          avgProcessingTime: 1.2
        }
      },
      'job_003': {
        id: 'job_003',
        type: 'sentiment_analysis',
        status: 'failed',
        priority: 'normal',
        queue: 'default',
        userId: 'user_456',
        title: 'Analyze Article Sentiment',
        description: 'Perform sentiment analysis on user comments',
        payload: {
          articleId: 'article_789',
          commentIds: ['comment_1', 'comment_2', 'comment_3'],
          analysisType: 'detailed'
        },
        result: null,
        progress: 0,
        createdAt: '2024-01-15T11:15:00Z',
        startedAt: '2024-01-15T11:15:05Z',
        completedAt: null,
        duration: null,
        attempts: 2,
        maxAttempts: 3,
        timeout: 180000,
        error: {
          code: 'API_RATE_LIMIT',
          message: 'Sentiment analysis API rate limit exceeded',
          timestamp: '2024-01-15T11:16:30Z',
          details: {
            rateLimitReset: '2024-01-15T11:20:00Z',
            requestsRemaining: 0,
            retryAfter: 210
          }
        },
        nextRetry: '2024-01-15T11:20:00Z',
        logs: [
          {
            timestamp: '2024-01-15T11:15:00Z',
            level: 'info',
            message: 'Job created and queued'
          },
          {
            timestamp: '2024-01-15T11:15:05Z',
            level: 'info',
            message: 'Started sentiment analysis'
          },
          {
            timestamp: '2024-01-15T11:15:10Z',
            level: 'error',
            message: 'API rate limit exceeded on first attempt'
          },
          {
            timestamp: '2024-01-15T11:16:00Z',
            level: 'info',
            message: 'Retrying sentiment analysis (attempt 2)'
          },
          {
            timestamp: '2024-01-15T11:16:30Z',
            level: 'error',
            message: 'API rate limit exceeded on second attempt'
          }
        ],
        metrics: {
          commentsToAnalyze: 3,
          commentsProcessed: 0,
          apiCalls: 2,
          failedCalls: 2
        }
      }
    };
    
    const job = mockJobs[jobId];
    
    if (!job) {
      return NextResponse.json(
        { error: 'Job not found' },
        { status: 404 }
      );
    }
    
    return NextResponse.json({ job });
  } catch (error) {
    console.error(`[/api/jobs/${params?.jobId}] Error:`, error);
    return NextResponse.json(
      { error: error.message },
      { status: 500 }
    );
  }
}

// PATCH /api/jobs/[jobId]
export async function PATCH(request, { params }) {
  try {
    const { jobId } = params;
    const body = await request.json();
    const { action, ...updateData } = body;
    
    if (!jobId) {
      return NextResponse.json(
        { error: 'Job ID is required' },
        { status: 400 }
      );
    }
    
    if (!action) {
      return NextResponse.json(
        { error: 'Action is required' },
        { status: 400 }
      );
    }
    
    // Validate action
    const validActions = ['cancel', 'retry', 'pause', 'resume', 'update_priority', 'reschedule'];
    if (!validActions.includes(action)) {
      return NextResponse.json(
        { error: `Invalid action. Must be one of: ${validActions.join(', ')}` },
        { status: 400 }
      );
    }
    
    // Mock job update logic
    let updatedJob = {
      id: jobId,
      updatedAt: new Date().toISOString()
    };
    
    switch (action) {
      case 'cancel':
        updatedJob = {
          ...updatedJob,
          status: 'cancelled',
          completedAt: new Date().toISOString(),
          progress: 0,
          error: {
            code: 'USER_CANCELLED',
            message: 'Job cancelled by user',
            timestamp: new Date().toISOString()
          }
        };
        break;
        
      case 'retry':
        if (!updateData.maxAttempts || updateData.maxAttempts < 1) {
          return NextResponse.json(
            { error: 'maxAttempts must be provided and greater than 0 for retry action' },
            { status: 400 }
          );
        }
        updatedJob = {
          ...updatedJob,
          status: 'pending',
          progress: 0,
          attempts: 0,
          maxAttempts: updateData.maxAttempts,
          error: null,
          startedAt: null,
          completedAt: null,
          duration: null,
          nextRetry: null
        };
        break;
        
      case 'pause':
        updatedJob = {
          ...updatedJob,
          status: 'paused',
          pausedAt: new Date().toISOString()
        };
        break;
        
      case 'resume':
        updatedJob = {
          ...updatedJob,
          status: 'running',
          resumedAt: new Date().toISOString(),
          pausedAt: null
        };
        break;
        
      case 'update_priority':
        const validPriorities = ['low', 'normal', 'high', 'urgent'];
        if (!updateData.priority || !validPriorities.includes(updateData.priority)) {
          return NextResponse.json(
            { error: `Priority must be one of: ${validPriorities.join(', ')}` },
            { status: 400 }
          );
        }
        updatedJob = {
          ...updatedJob,
          priority: updateData.priority
        };
        break;
        
      case 'reschedule':
        if (!updateData.scheduledFor) {
          return NextResponse.json(
            { error: 'scheduledFor is required for reschedule action' },
            { status: 400 }
          );
        }
        updatedJob = {
          ...updatedJob,
          status: 'scheduled',
          scheduledFor: updateData.scheduledFor,
          startedAt: null,
          completedAt: null
        };
        break;
    }
    
    console.log(`Job ${jobId} updated with action: ${action}`, updatedJob);
    
    // In real implementation, update job in database/queue
    
    return NextResponse.json({
      success: true,
      job: updatedJob,
      message: `Job ${action} completed successfully`
    });
  } catch (error) {
    console.error(`[/api/jobs/${params?.jobId}] PATCH Error:`, error);
    return NextResponse.json(
      { error: error.message },
      { status: 500 }
    );
  }
}

// DELETE /api/jobs/[jobId]
export async function DELETE(request, { params }) {
  try {
    const { jobId } = params;
    const { searchParams } = new URL(request.url);
    const force = searchParams.get('force') === 'true';
    
    if (!jobId) {
      return NextResponse.json(
        { error: 'Job ID is required' },
        { status: 400 }
      );
    }
    
    // Mock job status check
    const jobStatus = 'completed'; // In real implementation, fetch from database
    
    // Check if job can be deleted
    if (!force && (jobStatus === 'running' || jobStatus === 'pending')) {
      return NextResponse.json(
        { 
          error: 'Cannot delete running or pending job. Use force=true to override or cancel the job first.',
          jobStatus 
        },
        { status: 409 }
      );
    }
    
    console.log(`Deleting job ${jobId}${force ? ' (forced)' : ''}`);
    
    // In real implementation, delete job from database/queue
    
    return NextResponse.json({
      success: true,
      message: `Job ${jobId} deleted successfully`,
      deletedAt: new Date().toISOString()
    });
  } catch (error) {
    console.error(`[/api/jobs/${params?.jobId}] DELETE Error:`, error);
    return NextResponse.json(
      { error: error.message },
      { status: 500 }
    );
  }
}