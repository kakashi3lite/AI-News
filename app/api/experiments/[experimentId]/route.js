import { NextResponse } from 'next/server';

// GET /api/experiments/[experimentId]
export async function GET(request, { params }) {
  try {
    const { experimentId } = params;
    
    // Mock experiment data - in real implementation, fetch from database
    const experiments = {
      'exp_001': {
        id: 'exp_001',
        name: 'Article Card Layout A/B Test',
        description: 'Testing different article card layouts to improve click-through rates',
        category: 'ui',
        status: 'active',
        type: 'ab_test',
        startDate: new Date(Date.now() - 1000 * 60 * 60 * 24 * 7).toISOString(),
        endDate: new Date(Date.now() + 1000 * 60 * 60 * 24 * 7).toISOString(),
        targetSegments: ['new', 'returning'],
        variants: [
          {
            id: 'control',
            name: 'Current Layout',
            description: 'Existing article card design',
            allocation: 50,
            metrics: {
              participants: 1250,
              clickThroughRate: 0.12,
              conversionRate: 0.08,
              engagementTime: 145,
              bounceRate: 0.35
            }
          },
          {
            id: 'variant_a',
            name: 'Enhanced Layout',
            description: 'Larger images and improved typography',
            allocation: 50,
            metrics: {
              participants: 1180,
              clickThroughRate: 0.15,
              conversionRate: 0.11,
              engagementTime: 168,
              bounceRate: 0.28
            }
          }
        ],
        metrics: {
          totalParticipants: 2430,
          statisticalSignificance: 0.95,
          confidence: 95,
          expectedLift: 0.25,
          actualLift: 0.28,
          pValue: 0.02
        },
        hypothesis: 'Improved visual hierarchy will increase user engagement',
        successCriteria: 'CTR improvement > 20%',
        config: {
          trafficAllocation: 100,
          minimumSampleSize: 1000,
          maxDuration: 30,
          autoStop: true,
          significanceThreshold: 0.95
        },
        timeline: [
          {
            date: new Date(Date.now() - 1000 * 60 * 60 * 24 * 7).toISOString(),
            event: 'experiment_started',
            description: 'Experiment launched with 50/50 traffic split'
          },
          {
            date: new Date(Date.now() - 1000 * 60 * 60 * 24 * 5).toISOString(),
            event: 'milestone_reached',
            description: 'Reached 1000 participants milestone'
          },
          {
            date: new Date(Date.now() - 1000 * 60 * 60 * 24 * 2).toISOString(),
            event: 'significance_achieved',
            description: 'Statistical significance achieved (95% confidence)'
          }
        ],
        createdBy: 'user_123',
        lastUpdated: new Date(Date.now() - 1000 * 60 * 60 * 2).toISOString()
      },
      'exp_002': {
        id: 'exp_002',
        name: 'Personalization Algorithm V2',
        description: 'Testing new ML model for article recommendations',
        category: 'algorithm',
        status: 'active',
        type: 'multivariate',
        startDate: new Date(Date.now() - 1000 * 60 * 60 * 24 * 14).toISOString(),
        endDate: new Date(Date.now() + 1000 * 60 * 60 * 24 * 14).toISOString(),
        targetSegments: ['returning', 'power'],
        variants: [
          {
            id: 'current_algo',
            name: 'Current Algorithm',
            description: 'Existing recommendation system',
            allocation: 33,
            metrics: {
              participants: 890,
              clickThroughRate: 0.18,
              conversionRate: 0.14,
              engagementTime: 210
            }
          },
          {
            id: 'ml_v2',
            name: 'ML Model V2',
            description: 'Enhanced neural network with user behavior patterns',
            allocation: 33,
            metrics: {
              participants: 912,
              clickThroughRate: 0.22,
              conversionRate: 0.17,
              engagementTime: 245
            }
          },
          {
            id: 'hybrid',
            name: 'Hybrid Approach',
            description: 'Combination of collaborative and content-based filtering',
            allocation: 34,
            metrics: {
              participants: 923,
              clickThroughRate: 0.20,
              conversionRate: 0.16,
              engagementTime: 228
            }
          }
        ],
        metrics: {
          totalParticipants: 2725,
          statisticalSignificance: 0.98,
          confidence: 98,
          expectedLift: 0.15,
          actualLift: 0.22
        },
        hypothesis: 'Advanced ML model will improve recommendation relevance',
        successCriteria: 'Engagement time increase > 15%',
        createdBy: 'user_456',
        lastUpdated: new Date(Date.now() - 1000 * 60 * 60 * 1).toISOString()
      }
    };

    const experiment = experiments[experimentId];
    
    if (!experiment) {
      return NextResponse.json(
        { error: 'Experiment not found' },
        { status: 404 }
      );
    }

    return NextResponse.json(experiment);
  } catch (error) {
    console.error(`[/api/experiments/${params.experimentId}] Error:`, error);
    return NextResponse.json(
      { error: error.message },
      { status: 500 }
    );
  }
}

// PATCH /api/experiments/[experimentId]
export async function PATCH(request, { params }) {
  try {
    const { experimentId } = params;
    const body = await request.json();
    const { action, ...updateData } = body;

    // Handle different actions
    switch (action) {
      case 'start':
        // Start experiment
        const startResult = {
          id: experimentId,
          status: 'active',
          startDate: new Date().toISOString(),
          message: 'Experiment started successfully',
          participants: 0,
          variants: updateData.variants || []
        };
        
        console.log(`Starting experiment ${experimentId}:`, startResult);
        return NextResponse.json(startResult);

      case 'stop':
        // Stop experiment
        const stopResult = {
          id: experimentId,
          status: 'completed',
          endDate: new Date().toISOString(),
          message: 'Experiment stopped successfully',
          finalMetrics: {
            totalParticipants: Math.floor(Math.random() * 5000) + 1000,
            duration: Math.floor(Math.random() * 30) + 1,
            significanceAchieved: Math.random() > 0.3
          }
        };
        
        console.log(`Stopping experiment ${experimentId}:`, stopResult);
        return NextResponse.json(stopResult);

      case 'pause':
        // Pause experiment
        const pauseResult = {
          id: experimentId,
          status: 'paused',
          pausedAt: new Date().toISOString(),
          message: 'Experiment paused successfully'
        };
        
        console.log(`Pausing experiment ${experimentId}:`, pauseResult);
        return NextResponse.json(pauseResult);

      case 'resume':
        // Resume experiment
        const resumeResult = {
          id: experimentId,
          status: 'active',
          resumedAt: new Date().toISOString(),
          message: 'Experiment resumed successfully'
        };
        
        console.log(`Resuming experiment ${experimentId}:`, resumeResult);
        return NextResponse.json(resumeResult);

      case 'update':
        // Update experiment configuration
        const updateResult = {
          id: experimentId,
          ...updateData,
          lastUpdated: new Date().toISOString(),
          message: 'Experiment updated successfully'
        };
        
        console.log(`Updating experiment ${experimentId}:`, updateResult);
        return NextResponse.json(updateResult);

      default:
        return NextResponse.json(
          { error: 'Invalid action. Supported actions: start, stop, pause, resume, update' },
          { status: 400 }
        );
    }
  } catch (error) {
    console.error(`[/api/experiments/${params.experimentId}] PATCH Error:`, error);
    return NextResponse.json(
      { error: error.message },
      { status: 500 }
    );
  }
}

// DELETE /api/experiments/[experimentId]
export async function DELETE(request, { params }) {
  try {
    const { experimentId } = params;
    
    // In real implementation, check if experiment can be deleted
    // (e.g., not currently running, user has permissions, etc.)
    
    console.log(`Deleting experiment ${experimentId}`);
    
    return NextResponse.json({
      success: true,
      message: 'Experiment deleted successfully',
      deletedId: experimentId
    });
  } catch (error) {
    console.error(`[/api/experiments/${params.experimentId}] DELETE Error:`, error);
    return NextResponse.json(
      { error: error.message },
      { status: 500 }
    );
  }
}