import { NextResponse } from 'next/server';

// GET /api/experiments/[experimentId]/metrics
export async function GET(request, { params }) {
  try {
    const { experimentId } = params;
    const { searchParams } = new URL(request.url);
    const timeRange = searchParams.get('timeRange') || '7d'; // 1d, 7d, 30d
    const granularity = searchParams.get('granularity') || 'daily'; // hourly, daily, weekly
    const metric = searchParams.get('metric'); // specific metric to focus on
    
    // Generate mock time series data based on time range
    const generateTimeSeriesData = (days, metricName, baseValue, variance = 0.1) => {
      const data = [];
      const now = new Date();
      
      for (let i = days - 1; i >= 0; i--) {
        const date = new Date(now.getTime() - i * 24 * 60 * 60 * 1000);
        const randomFactor = 1 + (Math.random() - 0.5) * variance;
        const value = baseValue * randomFactor;
        
        data.push({
          date: date.toISOString().split('T')[0],
          timestamp: date.toISOString(),
          value: parseFloat(value.toFixed(4)),
          participants: Math.floor(Math.random() * 200) + 50
        });
      }
      
      return data;
    };
    
    const getDaysFromRange = (range) => {
      switch (range) {
        case '1d': return 1;
        case '7d': return 7;
        case '30d': return 30;
        default: return 7;
      }
    };
    
    const days = getDaysFromRange(timeRange);
    
    // Mock experiment metrics data
    const experimentMetrics = {
      'exp_001': {
        overview: {
          status: 'active',
          duration: 7,
          totalParticipants: 2430,
          variants: {
            control: {
              participants: 1250,
              conversionRate: 0.08,
              clickThroughRate: 0.12,
              engagementTime: 145,
              bounceRate: 0.35
            },
            variant_a: {
              participants: 1180,
              conversionRate: 0.11,
              clickThroughRate: 0.15,
              engagementTime: 168,
              bounceRate: 0.28
            }
          },
          significance: {
            conversionRate: { pValue: 0.02, significant: true, lift: 0.375 },
            clickThroughRate: { pValue: 0.01, significant: true, lift: 0.25 },
            engagementTime: { pValue: 0.03, significant: true, lift: 0.159 },
            bounceRate: { pValue: 0.04, significant: true, lift: -0.2 }
          }
        },
        timeSeries: {
          conversionRate: {
            control: generateTimeSeriesData(days, 'conversionRate', 0.08, 0.15),
            variant_a: generateTimeSeriesData(days, 'conversionRate', 0.11, 0.12)
          },
          clickThroughRate: {
            control: generateTimeSeriesData(days, 'clickThroughRate', 0.12, 0.2),
            variant_a: generateTimeSeriesData(days, 'clickThroughRate', 0.15, 0.18)
          },
          engagementTime: {
            control: generateTimeSeriesData(days, 'engagementTime', 145, 0.25),
            variant_a: generateTimeSeriesData(days, 'engagementTime', 168, 0.22)
          },
          participants: {
            control: generateTimeSeriesData(days, 'participants', 180, 0.3),
            variant_a: generateTimeSeriesData(days, 'participants', 170, 0.3)
          }
        },
        segmentBreakdown: {
          new_users: {
            control: { participants: 625, conversionRate: 0.06, clickThroughRate: 0.10 },
            variant_a: { participants: 590, conversionRate: 0.09, clickThroughRate: 0.13 }
          },
          returning_users: {
            control: { participants: 625, conversionRate: 0.10, clickThroughRate: 0.14 },
            variant_a: { participants: 590, conversionRate: 0.13, clickThroughRate: 0.17 }
          }
        },
        deviceBreakdown: {
          desktop: {
            control: { participants: 750, conversionRate: 0.09, clickThroughRate: 0.13 },
            variant_a: { participants: 708, conversionRate: 0.12, clickThroughRate: 0.16 }
          },
          mobile: {
            control: { participants: 375, conversionRate: 0.06, clickThroughRate: 0.10 },
            variant_a: { participants: 354, conversionRate: 0.09, clickThroughRate: 0.13 }
          },
          tablet: {
            control: { participants: 125, conversionRate: 0.08, clickThroughRate: 0.12 },
            variant_a: { participants: 118, conversionRate: 0.11, clickThroughRate: 0.15 }
          }
        }
      },
      'exp_002': {
        overview: {
          status: 'active',
          duration: 14,
          totalParticipants: 2725,
          variants: {
            current_algo: {
              participants: 890,
              conversionRate: 0.14,
              clickThroughRate: 0.18,
              engagementTime: 210
            },
            ml_v2: {
              participants: 912,
              conversionRate: 0.17,
              clickThroughRate: 0.22,
              engagementTime: 245
            },
            hybrid: {
              participants: 923,
              conversionRate: 0.16,
              clickThroughRate: 0.20,
              engagementTime: 228
            }
          },
          significance: {
            conversionRate: { pValue: 0.01, significant: true, lift: 0.214 },
            clickThroughRate: { pValue: 0.005, significant: true, lift: 0.222 },
            engagementTime: { pValue: 0.001, significant: true, lift: 0.167 }
          }
        },
        timeSeries: {
          conversionRate: {
            current_algo: generateTimeSeriesData(days, 'conversionRate', 0.14, 0.12),
            ml_v2: generateTimeSeriesData(days, 'conversionRate', 0.17, 0.10),
            hybrid: generateTimeSeriesData(days, 'conversionRate', 0.16, 0.11)
          },
          clickThroughRate: {
            current_algo: generateTimeSeriesData(days, 'clickThroughRate', 0.18, 0.15),
            ml_v2: generateTimeSeriesData(days, 'clickThroughRate', 0.22, 0.12),
            hybrid: generateTimeSeriesData(days, 'clickThroughRate', 0.20, 0.13)
          },
          engagementTime: {
            current_algo: generateTimeSeriesData(days, 'engagementTime', 210, 0.20),
            ml_v2: generateTimeSeriesData(days, 'engagementTime', 245, 0.18),
            hybrid: generateTimeSeriesData(days, 'engagementTime', 228, 0.19)
          }
        }
      }
    };
    
    const metrics = experimentMetrics[experimentId];
    
    if (!metrics) {
      return NextResponse.json(
        { error: 'Experiment metrics not found' },
        { status: 404 }
      );
    }
    
    // Filter by specific metric if requested
    if (metric && metrics.timeSeries[metric]) {
      return NextResponse.json({
        experimentId,
        metric,
        timeRange,
        granularity,
        data: metrics.timeSeries[metric],
        overview: metrics.overview.variants,
        significance: metrics.overview.significance[metric]
      });
    }
    
    // Return all metrics
    return NextResponse.json({
      experimentId,
      timeRange,
      granularity,
      ...metrics,
      generatedAt: new Date().toISOString()
    });
  } catch (error) {
    console.error(`[/api/experiments/${params.experimentId}/metrics] Error:`, error);
    return NextResponse.json(
      { error: error.message },
      { status: 500 }
    );
  }
}

// POST /api/experiments/[experimentId]/metrics
export async function POST(request, { params }) {
  try {
    const { experimentId } = params;
    const body = await request.json();
    const { event, variantId, userId, sessionId, metadata } = body;
    
    // Validate required fields
    if (!event || !variantId) {
      return NextResponse.json(
        { error: 'Missing required fields: event, variantId' },
        { status: 400 }
      );
    }
    
    // Mock tracking event
    const trackingEvent = {
      id: `event_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      experimentId,
      event,
      variantId,
      userId: userId || `anonymous_${Math.random().toString(36).substr(2, 9)}`,
      sessionId: sessionId || `session_${Math.random().toString(36).substr(2, 9)}`,
      timestamp: new Date().toISOString(),
      metadata: metadata || {},
      processed: false
    };
    
    console.log('Tracking experiment event:', trackingEvent);
    
    // In real implementation, save to analytics database
    // and update experiment metrics in real-time
    
    return NextResponse.json({
      success: true,
      eventId: trackingEvent.id,
      message: 'Event tracked successfully'
    });
  } catch (error) {
    console.error(`[/api/experiments/${params.experimentId}/metrics] POST Error:`, error);
    return NextResponse.json(
      { error: error.message },
      { status: 500 }
    );
  }
}