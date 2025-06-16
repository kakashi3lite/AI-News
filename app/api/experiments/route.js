import { NextResponse } from 'next/server';

// GET /api/experiments
export async function GET(request) {
  try {
    const { searchParams } = new URL(request.url);
    const status = searchParams.get('status'); // active, completed, draft
    const category = searchParams.get('category'); // ui, algorithm, feature
    const userSegment = searchParams.get('userSegment'); // new, returning, power
    const limit = parseInt(searchParams.get('limit')) || 20;
    const offset = parseInt(searchParams.get('offset')) || 0;
    
    // Mock experiments data
    const allExperiments = [
      {
        id: 'exp_001',
        name: 'Article Card Layout A/B Test',
        description: 'Testing different article card layouts to improve click-through rates',
        category: 'ui',
        status: 'active',
        type: 'ab_test',
        startDate: new Date(Date.now() - 1000 * 60 * 60 * 24 * 7).toISOString(), // 1 week ago
        endDate: new Date(Date.now() + 1000 * 60 * 60 * 24 * 7).toISOString(), // 1 week from now
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
              engagementTime: 145
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
              engagementTime: 168
            }
          }
        ],
        metrics: {
          totalParticipants: 2430,
          statisticalSignificance: 0.95,
          confidence: 95,
          expectedLift: 0.25,
          actualLift: 0.28
        },
        hypothesis: 'Improved visual hierarchy will increase user engagement',
        successCriteria: 'CTR improvement > 20%',
        createdBy: 'user_123',
        lastUpdated: new Date(Date.now() - 1000 * 60 * 60 * 2).toISOString()
      },
      {
        id: 'exp_002',
        name: 'Personalization Algorithm V2',
        description: 'Testing new ML model for article recommendations',
        category: 'algorithm',
        status: 'active',
        type: 'multivariate',
        startDate: new Date(Date.now() - 1000 * 60 * 60 * 24 * 14).toISOString(), // 2 weeks ago
        endDate: new Date(Date.now() + 1000 * 60 * 60 * 24 * 14).toISOString(), // 2 weeks from now
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
      },
      {
        id: 'exp_003',
        name: 'Dark Mode Feature Test',
        description: 'Testing user adoption and engagement with dark mode',
        category: 'feature',
        status: 'completed',
        type: 'feature_flag',
        startDate: new Date(Date.now() - 1000 * 60 * 60 * 24 * 30).toISOString(), // 30 days ago
        endDate: new Date(Date.now() - 1000 * 60 * 60 * 24 * 2).toISOString(), // 2 days ago
        targetSegments: ['new', 'returning', 'power'],
        variants: [
          {
            id: 'no_dark_mode',
            name: 'Light Mode Only',
            description: 'Standard light theme',
            allocation: 50,
            metrics: {
              participants: 1560,
              clickThroughRate: 0.14,
              conversionRate: 0.09,
              engagementTime: 180,
              sessionDuration: 420
            }
          },
          {
            id: 'with_dark_mode',
            name: 'Dark Mode Available',
            description: 'Option to switch to dark theme',
            allocation: 50,
            metrics: {
              participants: 1580,
              clickThroughRate: 0.16,
              conversionRate: 0.12,
              engagementTime: 205,
              sessionDuration: 485,
              darkModeAdoption: 0.68
            }
          }
        ],
        metrics: {
          totalParticipants: 3140,
          statisticalSignificance: 0.99,
          confidence: 99,
          expectedLift: 0.10,
          actualLift: 0.14
        },
        hypothesis: 'Dark mode option will increase user satisfaction and session duration',
        successCriteria: 'Session duration increase > 10%',
        results: {
          winner: 'with_dark_mode',
          summary: 'Dark mode feature significantly improved user engagement and session duration',
          recommendations: 'Roll out dark mode to all users'
        },
        createdBy: 'user_789',
        lastUpdated: new Date(Date.now() - 1000 * 60 * 60 * 24 * 2).toISOString()
      },
      {
        id: 'exp_004',
        name: 'Search Suggestions Enhancement',
        description: 'Testing improved search autocomplete functionality',
        category: 'feature',
        status: 'draft',
        type: 'ab_test',
        targetSegments: ['returning', 'power'],
        variants: [
          {
            id: 'current_search',
            name: 'Current Search',
            description: 'Existing search functionality',
            allocation: 50
          },
          {
            id: 'enhanced_search',
            name: 'Enhanced Suggestions',
            description: 'AI-powered search suggestions with context',
            allocation: 50
          }
        ],
        hypothesis: 'Better search suggestions will improve content discovery',
        successCriteria: 'Search success rate > 25% improvement',
        createdBy: 'user_101',
        lastUpdated: new Date(Date.now() - 1000 * 60 * 60 * 24).toISOString()
      },
      {
        id: 'exp_005',
        name: 'Social Sharing Optimization',
        description: 'Testing different social sharing button placements',
        category: 'ui',
        status: 'active',
        type: 'multivariate',
        startDate: new Date(Date.now() - 1000 * 60 * 60 * 24 * 3).toISOString(), // 3 days ago
        endDate: new Date(Date.now() + 1000 * 60 * 60 * 24 * 11).toISOString(), // 11 days from now
        targetSegments: ['new', 'returning'],
        variants: [
          {
            id: 'top_placement',
            name: 'Top Placement',
            description: 'Social buttons at top of article',
            allocation: 33,
            metrics: {
              participants: 456,
              clickThroughRate: 0.08,
              shareRate: 0.05,
              engagementTime: 125
            }
          },
          {
            id: 'bottom_placement',
            name: 'Bottom Placement',
            description: 'Social buttons at bottom of article',
            allocation: 33,
            metrics: {
              participants: 478,
              clickThroughRate: 0.06,
              shareRate: 0.08,
              engagementTime: 142
            }
          },
          {
            id: 'floating_placement',
            name: 'Floating Sidebar',
            description: 'Floating social buttons on side',
            allocation: 34,
            metrics: {
              participants: 467,
              clickThroughRate: 0.07,
              shareRate: 0.09,
              engagementTime: 138
            }
          }
        ],
        metrics: {
          totalParticipants: 1401,
          statisticalSignificance: 0.85,
          confidence: 85,
          expectedLift: 0.20,
          actualLift: 0.12
        },
        hypothesis: 'Optimal social button placement will increase sharing behavior',
        successCriteria: 'Share rate improvement > 20%',
        createdBy: 'user_202',
        lastUpdated: new Date(Date.now() - 1000 * 60 * 30).toISOString()
      }
    ];

    // Filter experiments based on query parameters
    let experiments = allExperiments;
    
    if (status) {
      experiments = experiments.filter(exp => exp.status === status);
    }
    
    if (category) {
      experiments = experiments.filter(exp => exp.category === category);
    }
    
    if (userSegment) {
      experiments = experiments.filter(exp => 
        exp.targetSegments && exp.targetSegments.includes(userSegment)
      );
    }

    // Apply pagination
    const paginatedExperiments = experiments.slice(offset, offset + limit);

    // Calculate summary statistics
    const stats = {
      total: experiments.length,
      byStatus: {
        active: experiments.filter(exp => exp.status === 'active').length,
        completed: experiments.filter(exp => exp.status === 'completed').length,
        draft: experiments.filter(exp => exp.status === 'draft').length
      },
      byCategory: {
        ui: experiments.filter(exp => exp.category === 'ui').length,
        algorithm: experiments.filter(exp => exp.category === 'algorithm').length,
        feature: experiments.filter(exp => exp.category === 'feature').length
      },
      totalParticipants: experiments
        .filter(exp => exp.metrics && exp.metrics.totalParticipants)
        .reduce((sum, exp) => sum + exp.metrics.totalParticipants, 0)
    };

    const response = {
      experiments: paginatedExperiments,
      stats,
      pagination: {
        total: experiments.length,
        limit,
        offset,
        hasMore: offset + limit < experiments.length
      }
    };

    return NextResponse.json(response);
  } catch (error) {
    console.error('[/api/experiments] Error:', error);
    return NextResponse.json(
      { error: error.message },
      { status: 500 }
    );
  }
}

// POST /api/experiments
export async function POST(request) {
  try {
    const body = await request.json();
    const {
      name,
      description,
      category,
      type,
      targetSegments,
      variants,
      hypothesis,
      successCriteria,
      duration,
      createdBy
    } = body;

    // Validate required fields
    if (!name || !description || !category || !type) {
      return NextResponse.json(
        { error: 'Missing required fields: name, description, category, type' },
        { status: 400 }
      );
    }

    // Generate new experiment ID
    const experimentId = `exp_${Date.now()}`;
    
    // Create new experiment
    const newExperiment = {
      id: experimentId,
      name,
      description,
      category,
      status: 'draft',
      type,
      targetSegments: targetSegments || ['new', 'returning'],
      variants: variants || [
        {
          id: 'control',
          name: 'Control',
          description: 'Current implementation',
          allocation: 50
        },
        {
          id: 'variant_a',
          name: 'Variant A',
          description: 'Test implementation',
          allocation: 50
        }
      ],
      hypothesis: hypothesis || '',
      successCriteria: successCriteria || '',
      duration: duration || 14, // days
      createdBy: createdBy || 'unknown',
      createdAt: new Date().toISOString(),
      lastUpdated: new Date().toISOString()
    };

    // In a real implementation, save to database
    console.log('Created new experiment:', newExperiment);

    return NextResponse.json({
      success: true,
      experiment: newExperiment,
      message: 'Experiment created successfully'
    }, { status: 201 });
  } catch (error) {
    console.error('[/api/experiments] POST Error:', error);
    return NextResponse.json(
      { error: error.message },
      { status: 500 }
    );
  }
}