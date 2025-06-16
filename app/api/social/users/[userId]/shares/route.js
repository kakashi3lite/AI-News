import { NextResponse } from 'next/server';

// GET /api/social/users/[userId]/shares
export async function GET(request, { params }) {
  try {
    const { userId } = params;
    const { searchParams } = new URL(request.url);
    const platform = searchParams.get('platform'); // twitter, facebook, linkedin, etc.
    const limit = parseInt(searchParams.get('limit')) || 20;
    const offset = parseInt(searchParams.get('offset')) || 0;
    const timeRange = searchParams.get('timeRange'); // day, week, month, all
    
    // Mock user shares data
    const allShares = [
      {
        id: 'share_1',
        articleId: 'ai-trends-2024',
        articleTitle: 'AI Trends in 2024: What to Expect',
        articleUrl: '/article/ai-trends-2024',
        platform: 'twitter',
        shareUrl: 'https://twitter.com/user/status/1234567890',
        timestamp: new Date(Date.now() - 1000 * 60 * 60 * 2).toISOString(), // 2 hours ago
        content: 'Fascinating insights into the future of AI! 🤖 This article covers everything from GPT advancements to robotics. #AI #Technology #Future',
        engagement: {
          likes: 15,
          retweets: 8,
          comments: 3,
          clicks: 45
        },
        visibility: 'public',
        tags: ['AI', 'Technology', 'Future']
      },
      {
        id: 'share_2',
        articleId: 'quantum-computing',
        articleTitle: 'Quantum Computing Breakthrough',
        articleUrl: '/article/quantum-computing',
        platform: 'linkedin',
        shareUrl: 'https://linkedin.com/posts/user_quantum_post',
        timestamp: new Date(Date.now() - 1000 * 60 * 60 * 24).toISOString(), // 1 day ago
        content: 'Major breakthrough in quantum computing! This could revolutionize how we approach complex computational problems. Excited to see where this leads.',
        engagement: {
          likes: 23,
          shares: 12,
          comments: 7,
          clicks: 89
        },
        visibility: 'public',
        tags: ['Quantum', 'Computing', 'Science']
      },
      {
        id: 'share_3',
        articleId: 'blockchain-healthcare',
        articleTitle: 'Blockchain in Healthcare: A Game Changer',
        articleUrl: '/article/blockchain-healthcare',
        platform: 'facebook',
        shareUrl: 'https://facebook.com/posts/blockchain_healthcare',
        timestamp: new Date(Date.now() - 1000 * 60 * 60 * 48).toISOString(), // 2 days ago
        content: 'Interesting perspective on how blockchain can transform healthcare data management and patient privacy.',
        engagement: {
          likes: 31,
          shares: 18,
          comments: 12,
          clicks: 67
        },
        visibility: 'public',
        tags: ['Blockchain', 'Healthcare', 'Privacy']
      },
      {
        id: 'share_4',
        articleId: 'sustainable-tech',
        articleTitle: 'Sustainable Technology Solutions',
        articleUrl: '/article/sustainable-tech',
        platform: 'twitter',
        shareUrl: 'https://twitter.com/user/status/9876543210',
        timestamp: new Date(Date.now() - 1000 * 60 * 60 * 72).toISOString(), // 3 days ago
        content: 'We need more initiatives like this! 🌱 Sustainability should be at the core of all tech development. Great read on eco-friendly innovations.',
        engagement: {
          likes: 42,
          retweets: 25,
          comments: 8,
          clicks: 123
        },
        visibility: 'public',
        tags: ['Sustainability', 'GreenTech', 'Environment']
      },
      {
        id: 'share_5',
        articleId: 'cybersecurity-trends',
        articleTitle: 'Cybersecurity Trends for 2024',
        articleUrl: '/article/cybersecurity-trends',
        platform: 'linkedin',
        shareUrl: 'https://linkedin.com/posts/user_cybersecurity',
        timestamp: new Date(Date.now() - 1000 * 60 * 60 * 24 * 7).toISOString(), // 1 week ago
        content: 'Essential reading for anyone in tech. The cybersecurity landscape is evolving rapidly, and staying informed is crucial.',
        engagement: {
          likes: 56,
          shares: 34,
          comments: 15,
          clicks: 178
        },
        visibility: 'public',
        tags: ['Cybersecurity', 'InfoSec', 'Technology']
      },
      {
        id: 'share_6',
        articleId: 'remote-work-future',
        articleTitle: 'The Future of Remote Work',
        articleUrl: '/article/remote-work-future',
        platform: 'internal',
        timestamp: new Date(Date.now() - 1000 * 60 * 60 * 24 * 14).toISOString(), // 2 weeks ago
        content: 'Shared with my team - great insights on how remote work is shaping the future of business.',
        engagement: {
          likes: 8,
          comments: 4,
          views: 23
        },
        visibility: 'team',
        tags: ['RemoteWork', 'Future', 'Business']
      }
    ];

    // Filter by platform if specified
    let shares = allShares;
    if (platform) {
      shares = shares.filter(share => share.platform === platform);
    }

    // Filter by time range if specified
    if (timeRange) {
      const now = Date.now();
      let cutoffTime;
      
      switch (timeRange) {
        case 'day':
          cutoffTime = now - 1000 * 60 * 60 * 24;
          break;
        case 'week':
          cutoffTime = now - 1000 * 60 * 60 * 24 * 7;
          break;
        case 'month':
          cutoffTime = now - 1000 * 60 * 60 * 24 * 30;
          break;
        default:
          cutoffTime = 0; // all time
      }
      
      shares = shares.filter(share => new Date(share.timestamp).getTime() > cutoffTime);
    }

    // Apply pagination
    const paginatedShares = shares.slice(offset, offset + limit);

    // Calculate engagement statistics
    const totalEngagement = shares.reduce((acc, share) => {
      acc.likes += share.engagement.likes || 0;
      acc.shares += share.engagement.shares || share.engagement.retweets || 0;
      acc.comments += share.engagement.comments || 0;
      acc.clicks += share.engagement.clicks || share.engagement.views || 0;
      return acc;
    }, { likes: 0, shares: 0, comments: 0, clicks: 0 });

    // Platform distribution
    const platformStats = shares.reduce((acc, share) => {
      acc[share.platform] = (acc[share.platform] || 0) + 1;
      return acc;
    }, {});

    // Top performing shares
    const topShares = [...shares]
      .sort((a, b) => {
        const aScore = (a.engagement.likes || 0) + (a.engagement.shares || a.engagement.retweets || 0) * 2 + (a.engagement.comments || 0) * 3;
        const bScore = (b.engagement.likes || 0) + (b.engagement.shares || b.engagement.retweets || 0) * 2 + (b.engagement.comments || 0) * 3;
        return bScore - aScore;
      })
      .slice(0, 3);

    const response = {
      shares: paginatedShares,
      stats: {
        total: shares.length,
        totalEngagement,
        platformDistribution: platformStats,
        topPerforming: topShares,
        averageEngagement: {
          likes: shares.length > 0 ? Math.round(totalEngagement.likes / shares.length) : 0,
          shares: shares.length > 0 ? Math.round(totalEngagement.shares / shares.length) : 0,
          comments: shares.length > 0 ? Math.round(totalEngagement.comments / shares.length) : 0,
          clicks: shares.length > 0 ? Math.round(totalEngagement.clicks / shares.length) : 0
        }
      },
      pagination: {
        total: shares.length,
        limit,
        offset,
        hasMore: offset + limit < shares.length
      }
    };

    return NextResponse.json(response);
  } catch (error) {
    console.error(`[/api/social/users/${params.userId}/shares] Error:`, error);
    return NextResponse.json(
      { error: error.message },
      { status: 500 }
    );
  }
}