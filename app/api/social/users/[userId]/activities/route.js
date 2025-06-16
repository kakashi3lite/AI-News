import { NextResponse } from 'next/server';

// GET /api/social/users/[userId]/activities
export async function GET(request, { params }) {
  try {
    const { userId } = params;
    const { searchParams } = new URL(request.url);
    const type = searchParams.get('type'); // comment, share, like, follow
    const limit = parseInt(searchParams.get('limit')) || 20;
    const offset = parseInt(searchParams.get('offset')) || 0;
    
    // Mock user activities
    const allActivities = [
      {
        id: 'activity_1',
        type: 'comment',
        description: 'Commented on "AI Trends in 2024"',
        timestamp: new Date(Date.now() - 1000 * 60 * 30).toISOString(),
        articleId: 'ai-trends-2024',
        articleTitle: 'AI Trends in 2024',
        content: 'Great insights on the future of AI! I particularly found the section on neural networks fascinating.',
        metadata: {
          commentId: 'comment_123',
          likes: 5,
          replies: 2
        }
      },
      {
        id: 'activity_2',
        type: 'share',
        description: 'Shared "The Future of Machine Learning"',
        timestamp: new Date(Date.now() - 1000 * 60 * 60 * 2).toISOString(),
        articleId: 'future-ml',
        articleTitle: 'The Future of Machine Learning',
        platform: 'twitter',
        content: 'Must-read article about ML advancements! 🤖 #MachineLearning #AI',
        metadata: {
          shareId: 'share_456',
          engagement: {
            likes: 15,
            retweets: 8,
            comments: 3
          }
        }
      },
      {
        id: 'activity_3',
        type: 'like',
        description: 'Liked "Quantum Computing Breakthrough"',
        timestamp: new Date(Date.now() - 1000 * 60 * 60 * 4).toISOString(),
        articleId: 'quantum-breakthrough',
        articleTitle: 'Quantum Computing Breakthrough',
        metadata: {
          reactionType: 'like'
        }
      },
      {
        id: 'activity_4',
        type: 'follow',
        description: 'Started following @techexpert',
        timestamp: new Date(Date.now() - 1000 * 60 * 60 * 6).toISOString(),
        targetUserId: 'techexpert',
        targetUserName: 'Tech Expert',
        targetUserAvatar: 'https://api.dicebear.com/7.x/avataaars/svg?seed=techexpert',
        metadata: {
          followType: 'user'
        }
      },
      {
        id: 'activity_5',
        type: 'share',
        description: 'Shared "Blockchain in Healthcare"',
        timestamp: new Date(Date.now() - 1000 * 60 * 60 * 12).toISOString(),
        articleId: 'blockchain-healthcare',
        articleTitle: 'Blockchain in Healthcare',
        platform: 'linkedin',
        content: 'Interesting perspective on how blockchain can revolutionize healthcare data management.',
        metadata: {
          shareId: 'share_789',
          engagement: {
            likes: 23,
            shares: 12,
            comments: 7
          }
        }
      },
      {
        id: 'activity_6',
        type: 'comment',
        description: 'Commented on "Sustainable Tech Solutions"',
        timestamp: new Date(Date.now() - 1000 * 60 * 60 * 18).toISOString(),
        articleId: 'sustainable-tech',
        articleTitle: 'Sustainable Tech Solutions',
        content: 'We need more initiatives like this. Sustainability should be at the core of all tech development.',
        metadata: {
          commentId: 'comment_321',
          likes: 12,
          replies: 5
        }
      },
      {
        id: 'activity_7',
        type: 'like',
        description: 'Reacted with ❤️ to "Open Source AI Models"',
        timestamp: new Date(Date.now() - 1000 * 60 * 60 * 24).toISOString(),
        articleId: 'open-source-ai',
        articleTitle: 'Open Source AI Models',
        metadata: {
          reactionType: 'love'
        }
      },
      {
        id: 'activity_8',
        type: 'follow',
        description: 'Joined group "AI Researchers"',
        timestamp: new Date(Date.now() - 1000 * 60 * 60 * 36).toISOString(),
        targetGroupId: 'ai-researchers',
        targetGroupName: 'AI Researchers',
        metadata: {
          followType: 'group',
          memberCount: 1247
        }
      }
    ];

    // Filter by type if specified
    let activities = allActivities;
    if (type) {
      activities = activities.filter(activity => activity.type === type);
    }

    // Apply pagination
    const paginatedActivities = activities.slice(offset, offset + limit);

    // Add activity statistics
    const stats = {
      total: activities.length,
      byType: {
        comment: allActivities.filter(a => a.type === 'comment').length,
        share: allActivities.filter(a => a.type === 'share').length,
        like: allActivities.filter(a => a.type === 'like').length,
        follow: allActivities.filter(a => a.type === 'follow').length
      },
      timeRange: {
        last24h: allActivities.filter(a => 
          new Date(a.timestamp) > new Date(Date.now() - 1000 * 60 * 60 * 24)
        ).length,
        lastWeek: allActivities.filter(a => 
          new Date(a.timestamp) > new Date(Date.now() - 1000 * 60 * 60 * 24 * 7)
        ).length,
        lastMonth: allActivities.filter(a => 
          new Date(a.timestamp) > new Date(Date.now() - 1000 * 60 * 60 * 24 * 30)
        ).length
      }
    };

    const response = {
      activities: paginatedActivities,
      stats,
      pagination: {
        total: activities.length,
        limit,
        offset,
        hasMore: offset + limit < activities.length
      }
    };

    return NextResponse.json(response);
  } catch (error) {
    console.error(`[/api/social/users/${params.userId}/activities] Error:`, error);
    return NextResponse.json(
      { error: error.message },
      { status: 500 }
    );
  }
}