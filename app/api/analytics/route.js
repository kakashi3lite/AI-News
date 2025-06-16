import { NextResponse } from 'next/server';

// GET /api/analytics
export async function GET(request) {
  try {
    const { searchParams } = new URL(request.url);
    const timeRange = searchParams.get('timeRange') || '7d'; // 1d, 7d, 30d, 90d
    const metric = searchParams.get('metric'); // pageviews, users, sessions, etc.
    const segment = searchParams.get('segment'); // user segment filter
    const breakdown = searchParams.get('breakdown'); // device, location, source
    const detailed = searchParams.get('detailed') === 'true';
    
    // Generate mock time series data
    const generateTimeSeriesData = (days, baseValue, variance = 0.2, trend = 0) => {
      const data = [];
      const now = new Date();
      
      for (let i = days - 1; i >= 0; i--) {
        const date = new Date(now.getTime() - i * 24 * 60 * 60 * 1000);
        const dayOfWeek = date.getDay();
        const isWeekend = dayOfWeek === 0 || dayOfWeek === 6;
        const weekendFactor = isWeekend ? 0.7 : 1.0; // Lower traffic on weekends
        
        const trendValue = baseValue + (trend * (days - i - 1) / days);
        const randomFactor = 1 + (Math.random() - 0.5) * variance;
        const value = Math.max(0, Math.round(trendValue * weekendFactor * randomFactor));
        
        data.push({
          date: date.toISOString().split('T')[0],
          timestamp: date.toISOString(),
          value
        });
      }
      
      return data;
    };
    
    const getDaysFromRange = (range) => {
      switch (range) {
        case '1d': return 1;
        case '7d': return 7;
        case '30d': return 30;
        case '90d': return 90;
        default: return 7;
      }
    };
    
    const days = getDaysFromRange(timeRange);
    
    // Current period metrics
    const currentMetrics = {
      overview: {
        pageViews: 45678,
        uniqueVisitors: 12345,
        sessions: 18567,
        bounceRate: 0.32,
        avgSessionDuration: 8.5,
        pagesPerSession: 4.2,
        conversionRate: 0.034,
        newUsers: 0.28
      },
      traffic: {
        organic: 15234,
        direct: 12456,
        social: 8901,
        referral: 5678,
        email: 2345,
        paid: 1064
      },
      devices: {
        desktop: 0.52,
        mobile: 0.38,
        tablet: 0.10
      },
      browsers: {
        chrome: 0.65,
        firefox: 0.18,
        safari: 0.12,
        edge: 0.04,
        other: 0.01
      },
      geography: {
        'United States': 0.35,
        'United Kingdom': 0.12,
        'Germany': 0.08,
        'France': 0.07,
        'Canada': 0.06,
        'Australia': 0.05,
        'Japan': 0.04,
        'Other': 0.23
      },
      topPages: [
        { path: '/', views: 8945, uniqueViews: 6234, avgTime: 145 },
        { path: '/news/ai-trends-2024', views: 5678, uniqueViews: 4123, avgTime: 320 },
        { path: '/news/quantum-computing', views: 4321, uniqueViews: 3456, avgTime: 280 },
        { path: '/search', views: 3456, uniqueViews: 2789, avgTime: 95 },
        { path: '/profile', views: 2345, uniqueViews: 1890, avgTime: 180 }
      ],
      events: {
        articleRead: 23456,
        articleShared: 1234,
        searchPerformed: 5678,
        profileUpdated: 890,
        commentPosted: 567,
        reactionAdded: 2345
      }
    };
    
    // Time series data
    const timeSeriesData = {
      pageViews: generateTimeSeriesData(days, 1800, 0.25, 50),
      uniqueVisitors: generateTimeSeriesData(days, 650, 0.3, 20),
      sessions: generateTimeSeriesData(days, 920, 0.28, 30),
      bounceRate: generateTimeSeriesData(days, 32, 0.15).map(item => ({
        ...item,
        value: Math.min(100, Math.max(0, item.value))
      })),
      avgSessionDuration: generateTimeSeriesData(days, 8.5, 0.2).map(item => ({
        ...item,
        value: Math.max(0, parseFloat(item.value.toFixed(1)))
      })),
      conversionRate: generateTimeSeriesData(days, 3.4, 0.3).map(item => ({
        ...item,
        value: Math.max(0, parseFloat((item.value / 100).toFixed(4)))
      }))
    };
    
    // User segments analysis
    const userSegments = {
      new: {
        count: 3456,
        percentage: 0.28,
        avgSessionDuration: 6.2,
        bounceRate: 0.45,
        conversionRate: 0.021,
        topSources: ['organic', 'social', 'direct']
      },
      returning: {
        count: 7890,
        percentage: 0.64,
        avgSessionDuration: 9.8,
        bounceRate: 0.25,
        conversionRate: 0.042,
        topSources: ['direct', 'organic', 'email']
      },
      power: {
        count: 999,
        percentage: 0.08,
        avgSessionDuration: 15.6,
        bounceRate: 0.12,
        conversionRate: 0.089,
        topSources: ['direct', 'email', 'referral']
      }
    };
    
    // Real-time metrics (last hour)
    const realTime = {
      activeUsers: 234,
      pageViewsLastHour: 1456,
      topActivePages: [
        { path: '/', activeUsers: 45 },
        { path: '/news/breaking-ai-news', activeUsers: 32 },
        { path: '/search', activeUsers: 28 },
        { path: '/trending', activeUsers: 21 },
        { path: '/profile', activeUsers: 18 }
      ],
      trafficSources: {
        organic: 89,
        direct: 67,
        social: 45,
        referral: 23,
        email: 10
      },
      events: {
        articleRead: 156,
        searchPerformed: 89,
        articleShared: 23,
        commentPosted: 12,
        reactionAdded: 45
      }
    };
    
    // Performance metrics
    const performance = {
      pageLoadTime: {
        avg: 2.3,
        p50: 1.8,
        p95: 4.2,
        p99: 6.8
      },
      timeToInteractive: {
        avg: 3.1,
        p50: 2.5,
        p95: 5.8,
        p99: 8.9
      },
      coreWebVitals: {
        lcp: 2.1, // Largest Contentful Paint
        fid: 45, // First Input Delay (ms)
        cls: 0.08 // Cumulative Layout Shift
      },
      errorRate: 0.012,
      uptimePercentage: 99.95
    };
    
    // Build response based on filters
    let response = {
      timeRange,
      generatedAt: new Date().toISOString(),
      current: currentMetrics,
      realTime,
      performance
    };
    
    if (detailed) {
      response.timeSeries = timeSeriesData;
      response.userSegments = userSegments;
    }
    
    if (metric && timeSeriesData[metric]) {
      response.timeSeries = { [metric]: timeSeriesData[metric] };
    }
    
    if (segment && userSegments[segment]) {
      response.segment = userSegments[segment];
    }
    
    if (breakdown) {
      switch (breakdown) {
        case 'device':
          response.breakdown = currentMetrics.devices;
          break;
        case 'location':
          response.breakdown = currentMetrics.geography;
          break;
        case 'source':
          response.breakdown = currentMetrics.traffic;
          break;
        case 'browser':
          response.breakdown = currentMetrics.browsers;
          break;
      }
    }
    
    return NextResponse.json(response);
  } catch (error) {
    console.error('[/api/analytics] Error:', error);
    return NextResponse.json(
      { error: error.message },
      { status: 500 }
    );
  }
}

// POST /api/analytics
export async function POST(request) {
  try {
    const body = await request.json();
    const {
      event,
      userId,
      sessionId,
      page,
      properties,
      timestamp,
      userAgent,
      ip
    } = body;
    
    // Validate required fields
    if (!event) {
      return NextResponse.json(
        { error: 'Missing required field: event' },
        { status: 400 }
      );
    }
    
    // Create analytics event
    const analyticsEvent = {
      id: `event_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      event,
      userId: userId || `anonymous_${Math.random().toString(36).substr(2, 9)}`,
      sessionId: sessionId || `session_${Math.random().toString(36).substr(2, 9)}`,
      page: page || request.headers.get('referer'),
      properties: properties || {},
      timestamp: timestamp || new Date().toISOString(),
      userAgent: userAgent || request.headers.get('user-agent'),
      ip: ip || request.headers.get('x-forwarded-for') || 'unknown',
      processed: false
    };
    
    console.log('Tracking analytics event:', analyticsEvent);
    
    // In real implementation, save to analytics database
    // and update real-time metrics
    
    return NextResponse.json({
      success: true,
      eventId: analyticsEvent.id,
      message: 'Event tracked successfully'
    });
  } catch (error) {
    console.error('[/api/analytics] POST Error:', error);
    return NextResponse.json(
      { error: error.message },
      { status: 500 }
    );
  }
}