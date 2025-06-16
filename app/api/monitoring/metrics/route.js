import { NextResponse } from 'next/server';

// GET /api/monitoring/metrics
export async function GET(request) {
  try {
    const { searchParams } = new URL(request.url);
    const timeRange = searchParams.get('timeRange') || '1h'; // 1h, 6h, 24h, 7d
    const category = searchParams.get('category'); // system, performance, user, business
    const detailed = searchParams.get('detailed') === 'true';
    
    // Generate mock time series data
    const generateMetricData = (points, baseValue, variance = 0.1, trend = 0) => {
      const data = [];
      const now = Date.now();
      const interval = getIntervalFromRange(timeRange);
      
      for (let i = points - 1; i >= 0; i--) {
        const timestamp = new Date(now - i * interval);
        const trendValue = baseValue + (trend * (points - i - 1) / points);
        const randomFactor = 1 + (Math.random() - 0.5) * variance;
        const value = Math.max(0, trendValue * randomFactor);
        
        data.push({
          timestamp: timestamp.toISOString(),
          value: parseFloat(value.toFixed(2))
        });
      }
      
      return data;
    };
    
    const getIntervalFromRange = (range) => {
      switch (range) {
        case '1h': return 60 * 1000; // 1 minute intervals
        case '6h': return 5 * 60 * 1000; // 5 minute intervals
        case '24h': return 15 * 60 * 1000; // 15 minute intervals
        case '7d': return 60 * 60 * 1000; // 1 hour intervals
        default: return 60 * 1000;
      }
    };
    
    const getPointsFromRange = (range) => {
      switch (range) {
        case '1h': return 60;
        case '6h': return 72;
        case '24h': return 96;
        case '7d': return 168;
        default: return 60;
      }
    };
    
    const points = getPointsFromRange(timeRange);
    
    // Current system metrics
    const currentMetrics = {
      system: {
        cpu: {
          usage: 45.2,
          cores: 8,
          loadAverage: [1.2, 1.5, 1.8],
          temperature: 65
        },
        memory: {
          used: 6.8,
          total: 16.0,
          usage: 42.5,
          available: 9.2,
          cached: 2.1
        },
        disk: {
          used: 120.5,
          total: 500.0,
          usage: 24.1,
          available: 379.5,
          iops: 1250
        },
        network: {
          inbound: 125.6,
          outbound: 89.3,
          connections: 342,
          bandwidth: 1000
        }
      },
      performance: {
        responseTime: {
          avg: 245,
          p50: 180,
          p95: 450,
          p99: 890
        },
        throughput: {
          requestsPerSecond: 125.6,
          requestsPerMinute: 7536,
          peakRps: 189.2
        },
        errors: {
          rate: 0.8,
          count: 23,
          types: {
            '4xx': 18,
            '5xx': 5
          }
        },
        availability: {
          uptime: 99.95,
          downtime: 0.05,
          incidents: 1
        }
      },
      user: {
        active: {
          current: 1247,
          peak: 1892,
          average: 1156
        },
        sessions: {
          total: 3456,
          new: 892,
          returning: 2564,
          avgDuration: 8.5
        },
        engagement: {
          pageViews: 15678,
          bounceRate: 0.32,
          avgSessionDuration: 8.5,
          pagesPerSession: 4.2
        },
        geography: {
          'US': 45.2,
          'EU': 28.7,
          'ASIA': 18.9,
          'OTHER': 7.2
        }
      },
      business: {
        revenue: {
          current: 12450.67,
          target: 15000.00,
          growth: 8.5
        },
        conversions: {
          rate: 3.2,
          count: 156,
          value: 89.50
        },
        retention: {
          daily: 0.68,
          weekly: 0.45,
          monthly: 0.23
        }
      }
    };
    
    // Time series data
    const timeSeriesData = {
      system: {
        cpuUsage: generateMetricData(points, 45, 0.2),
        memoryUsage: generateMetricData(points, 42, 0.15),
        diskUsage: generateMetricData(points, 24, 0.1),
        networkIn: generateMetricData(points, 125, 0.3),
        networkOut: generateMetricData(points, 89, 0.25)
      },
      performance: {
        responseTime: generateMetricData(points, 245, 0.4),
        throughput: generateMetricData(points, 125, 0.3),
        errorRate: generateMetricData(points, 0.8, 0.5),
        availability: generateMetricData(points, 99.95, 0.001)
      },
      user: {
        activeUsers: generateMetricData(points, 1247, 0.2),
        newSessions: generateMetricData(points, 892, 0.3),
        pageViews: generateMetricData(points, 15678, 0.25),
        bounceRate: generateMetricData(points, 32, 0.15)
      },
      business: {
        revenue: generateMetricData(points, 12450, 0.1, 500),
        conversions: generateMetricData(points, 156, 0.2),
        conversionRate: generateMetricData(points, 3.2, 0.15)
      }
    };
    
    // Alerts and incidents
    const alerts = [
      {
        id: 'alert_001',
        type: 'warning',
        severity: 'medium',
        title: 'High Memory Usage',
        description: 'Memory usage has exceeded 80% for the last 10 minutes',
        metric: 'memory.usage',
        threshold: 80,
        currentValue: 85.2,
        timestamp: new Date(Date.now() - 1000 * 60 * 8).toISOString(),
        status: 'active',
        acknowledged: false
      },
      {
        id: 'alert_002',
        type: 'error',
        severity: 'high',
        title: 'API Response Time Spike',
        description: 'API response time has increased by 150% in the last 5 minutes',
        metric: 'performance.responseTime',
        threshold: 500,
        currentValue: 890,
        timestamp: new Date(Date.now() - 1000 * 60 * 3).toISOString(),
        status: 'active',
        acknowledged: true,
        acknowledgedBy: 'admin@example.com'
      },
      {
        id: 'alert_003',
        type: 'info',
        severity: 'low',
        title: 'Disk Space Warning',
        description: 'Disk usage approaching 30% threshold',
        metric: 'disk.usage',
        threshold: 30,
        currentValue: 28.5,
        timestamp: new Date(Date.now() - 1000 * 60 * 15).toISOString(),
        status: 'resolved',
        acknowledged: true,
        resolvedAt: new Date(Date.now() - 1000 * 60 * 2).toISOString()
      }
    ];
    
    // Service health status
    const services = {
      api: {
        status: 'healthy',
        responseTime: 245,
        uptime: 99.95,
        lastCheck: new Date().toISOString()
      },
      database: {
        status: 'healthy',
        responseTime: 12,
        uptime: 99.99,
        connections: 45,
        lastCheck: new Date().toISOString()
      },
      cache: {
        status: 'warning',
        responseTime: 8,
        uptime: 99.8,
        hitRate: 0.85,
        lastCheck: new Date().toISOString()
      },
      search: {
        status: 'healthy',
        responseTime: 89,
        uptime: 99.92,
        indexSize: '2.3GB',
        lastCheck: new Date().toISOString()
      },
      cdn: {
        status: 'healthy',
        responseTime: 45,
        uptime: 99.98,
        bandwidth: '125MB/s',
        lastCheck: new Date().toISOString()
      }
    };
    
    // Build response based on category filter
    let response = {
      timestamp: new Date().toISOString(),
      timeRange,
      current: currentMetrics,
      alerts: alerts.filter(alert => alert.status === 'active'),
      services
    };
    
    if (detailed) {
      response.timeSeries = timeSeriesData;
      response.allAlerts = alerts;
    }
    
    if (category) {
      response.current = { [category]: currentMetrics[category] };
      if (detailed && timeSeriesData[category]) {
        response.timeSeries = { [category]: timeSeriesData[category] };
      }
    }
    
    return NextResponse.json(response);
  } catch (error) {
    console.error('[/api/monitoring/metrics] Error:', error);
    return NextResponse.json(
      { error: error.message },
      { status: 500 }
    );
  }
}

// POST /api/monitoring/metrics
export async function POST(request) {
  try {
    const body = await request.json();
    const { metric, value, timestamp, tags, source } = body;
    
    // Validate required fields
    if (!metric || value === undefined) {
      return NextResponse.json(
        { error: 'Missing required fields: metric, value' },
        { status: 400 }
      );
    }
    
    // Mock metric ingestion
    const metricEntry = {
      id: `metric_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      metric,
      value,
      timestamp: timestamp || new Date().toISOString(),
      tags: tags || {},
      source: source || 'api',
      processed: false
    };
    
    console.log('Ingesting metric:', metricEntry);
    
    // In real implementation, save to time series database
    // and trigger alert evaluation
    
    return NextResponse.json({
      success: true,
      metricId: metricEntry.id,
      message: 'Metric ingested successfully'
    });
  } catch (error) {
    console.error('[/api/monitoring/metrics] POST Error:', error);
    return NextResponse.json(
      { error: error.message },
      { status: 500 }
    );
  }
}