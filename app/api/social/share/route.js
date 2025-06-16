import { NextResponse } from 'next/server';

// POST /api/social/share
export async function POST(request) {
  try {
    const { articleId, platform, userId, message, visibility } = await request.json();
    
    if (!articleId || !platform || !userId) {
      return NextResponse.json(
        { error: 'articleId, platform, and userId are required' },
        { status: 400 }
      );
    }

    const validPlatforms = ['twitter', 'facebook', 'linkedin', 'reddit', 'email', 'copy', 'internal'];
    if (!validPlatforms.includes(platform)) {
      return NextResponse.json(
        { error: 'Invalid platform' },
        { status: 400 }
      );
    }

    console.log(`[/api/social/share] Sharing article ${articleId} to ${platform}`);
    
    // Mock share data
    const shareData = {
      id: Date.now().toString(),
      articleId,
      platform,
      userId,
      message: message || '',
      visibility: visibility || 'public',
      timestamp: new Date().toISOString(),
      shareUrl: `https://news-dashboard.com/article/${articleId}?ref=${platform}`,
      analytics: {
        clicks: 0,
        impressions: 0,
        engagement: 0
      }
    };

    // Platform-specific response data
    let platformResponse = {};
    
    switch (platform) {
      case 'twitter':
        platformResponse = {
          tweetId: `tweet_${Date.now()}`,
          tweetUrl: `https://twitter.com/user/status/${Date.now()}`,
          hashtags: ['#AI', '#News', '#Tech']
        };
        break;
      case 'facebook':
        platformResponse = {
          postId: `fb_${Date.now()}`,
          postUrl: `https://facebook.com/posts/${Date.now()}`
        };
        break;
      case 'linkedin':
        platformResponse = {
          postId: `li_${Date.now()}`,
          postUrl: `https://linkedin.com/posts/${Date.now()}`
        };
        break;
      case 'reddit':
        platformResponse = {
          postId: `reddit_${Date.now()}`,
          subreddit: 'technology',
          postUrl: `https://reddit.com/r/technology/comments/${Date.now()}`
        };
        break;
      case 'email':
        platformResponse = {
          emailId: `email_${Date.now()}`,
          subject: `Check out this article: ${shareData.shareUrl}`,
          recipients: 1
        };
        break;
      case 'copy':
        platformResponse = {
          copiedUrl: shareData.shareUrl,
          copyTimestamp: new Date().toISOString()
        };
        break;
      case 'internal':
        platformResponse = {
          internalId: `internal_${Date.now()}`,
          visibility: visibility || 'followers'
        };
        break;
    }

    const response = {
      success: true,
      share: shareData,
      platform: platformResponse,
      message: `Successfully shared to ${platform}`,
      shareCount: Math.floor(Math.random() * 50) + 1 // Mock updated share count
    };

    return NextResponse.json(response);
  } catch (error) {
    console.error('[/api/social/share] Error:', error);
    return NextResponse.json(
      { error: error.message },
      { status: 500 }
    );
  }
}

// GET /api/social/share
export async function GET(request) {
  try {
    const { searchParams } = new URL(request.url);
    const articleId = searchParams.get('articleId');
    const userId = searchParams.get('userId');
    
    if (!articleId) {
      return NextResponse.json(
        { error: 'articleId is required' },
        { status: 400 }
      );
    }

    // Mock share analytics
    const shareAnalytics = {
      articleId,
      totalShares: Math.floor(Math.random() * 100) + 10,
      platforms: {
        twitter: Math.floor(Math.random() * 30) + 5,
        facebook: Math.floor(Math.random() * 25) + 3,
        linkedin: Math.floor(Math.random() * 20) + 2,
        reddit: Math.floor(Math.random() * 15) + 1,
        email: Math.floor(Math.random() * 10) + 1,
        internal: Math.floor(Math.random() * 40) + 8
      },
      recentShares: [
        {
          platform: 'twitter',
          timestamp: new Date(Date.now() - 1000 * 60 * 30).toISOString(),
          userId: 'user123'
        },
        {
          platform: 'linkedin',
          timestamp: new Date(Date.now() - 1000 * 60 * 60 * 2).toISOString(),
          userId: 'user456'
        }
      ],
      userShares: userId ? [
        {
          platform: 'twitter',
          timestamp: new Date(Date.now() - 1000 * 60 * 60 * 24).toISOString()
        }
      ] : []
    };

    return NextResponse.json(shareAnalytics);
  } catch (error) {
    console.error('[/api/social/share] Error:', error);
    return NextResponse.json(
      { error: error.message },
      { status: 500 }
    );
  }
}