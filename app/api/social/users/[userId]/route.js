import { NextResponse } from 'next/server';

// GET /api/social/users/[userId]
export async function GET(request, { params }) {
  try {
    const { userId } = params;
    const { searchParams } = new URL(request.url);
    const include = searchParams.get('include'); // activities, shares, followers, etc.
    
    // Mock user profile data
    const userProfile = {
      id: userId,
      username: `user_${userId}`,
      displayName: 'John Doe',
      email: 'john.doe@example.com',
      avatar: `https://api.dicebear.com/7.x/avataaars/svg?seed=${userId}`,
      bio: 'Tech enthusiast and AI researcher. Passionate about the future of technology.',
      location: 'San Francisco, CA',
      website: 'https://johndoe.tech',
      joinedAt: new Date(Date.now() - 1000 * 60 * 60 * 24 * 365).toISOString(), // 1 year ago
      lastActive: new Date(Date.now() - 1000 * 60 * 30).toISOString(), // 30 min ago
      verified: Math.random() > 0.7,
      stats: {
        followers: Math.floor(Math.random() * 1000) + 100,
        following: Math.floor(Math.random() * 500) + 50,
        posts: Math.floor(Math.random() * 200) + 20,
        likes: Math.floor(Math.random() * 5000) + 500,
        shares: Math.floor(Math.random() * 1000) + 100,
        comments: Math.floor(Math.random() * 2000) + 200
      },
      preferences: {
        theme: 'dark',
        language: 'en',
        timezone: 'America/Los_Angeles',
        notifications: {
          email: true,
          push: false,
          inApp: true
        }
      },
      badges: [
        { id: 'early_adopter', name: 'Early Adopter', icon: '🌟' },
        { id: 'ai_expert', name: 'AI Expert', icon: '🤖' },
        { id: 'top_contributor', name: 'Top Contributor', icon: '🏆' }
      ],
      interests: ['AI', 'Machine Learning', 'Technology', 'Startups', 'Innovation'],
      socialLinks: {
        twitter: '@johndoe',
        linkedin: 'johndoe',
        github: 'johndoe'
      }
    };

    // Add additional data based on include parameter
    if (include) {
      const includeList = include.split(',');
      
      if (includeList.includes('activities')) {
        userProfile.recentActivities = [
          {
            id: 'activity_1',
            type: 'comment',
            description: 'Commented on "AI Trends in 2024"',
            timestamp: new Date(Date.now() - 1000 * 60 * 30).toISOString(),
            articleId: 'ai-trends-2024'
          },
          {
            id: 'activity_2',
            type: 'share',
            description: 'Shared "The Future of Machine Learning"',
            timestamp: new Date(Date.now() - 1000 * 60 * 60 * 2).toISOString(),
            articleId: 'future-ml'
          },
          {
            id: 'activity_3',
            type: 'follow',
            description: 'Started following @techexpert',
            timestamp: new Date(Date.now() - 1000 * 60 * 60 * 6).toISOString(),
            targetUserId: 'techexpert'
          }
        ];
      }
      
      if (includeList.includes('shares')) {
        userProfile.recentShares = [
          {
            id: 'share_1',
            articleId: 'ai-trends-2024',
            title: 'AI Trends in 2024',
            platform: 'twitter',
            timestamp: new Date(Date.now() - 1000 * 60 * 60 * 24).toISOString(),
            engagement: {
              likes: 15,
              retweets: 8,
              comments: 3
            }
          },
          {
            id: 'share_2',
            articleId: 'future-ml',
            title: 'The Future of Machine Learning',
            platform: 'linkedin',
            timestamp: new Date(Date.now() - 1000 * 60 * 60 * 48).toISOString(),
            engagement: {
              likes: 23,
              shares: 12,
              comments: 7
            }
          }
        ];
      }
      
      if (includeList.includes('followers')) {
        userProfile.followers = [
          {
            id: 'follower_1',
            username: 'techfan',
            displayName: 'Tech Fan',
            avatar: 'https://api.dicebear.com/7.x/avataaars/svg?seed=techfan',
            followedAt: new Date(Date.now() - 1000 * 60 * 60 * 24 * 7).toISOString()
          },
          {
            id: 'follower_2',
            username: 'airesearcher',
            displayName: 'AI Researcher',
            avatar: 'https://api.dicebear.com/7.x/avataaars/svg?seed=airesearcher',
            followedAt: new Date(Date.now() - 1000 * 60 * 60 * 24 * 14).toISOString()
          }
        ];
      }
    }

    return NextResponse.json(userProfile);
  } catch (error) {
    console.error(`[/api/social/users/${params.userId}] Error:`, error);
    return NextResponse.json(
      { error: error.message },
      { status: 500 }
    );
  }
}

// PATCH /api/social/users/[userId]
export async function PATCH(request, { params }) {
  try {
    const { userId } = params;
    const updates = await request.json();
    
    console.log(`[/api/social/users/${userId}] Updating profile:`, Object.keys(updates));
    
    // Mock updating user profile
    const updatedProfile = {
      id: userId,
      ...updates,
      updatedAt: new Date().toISOString()
    };

    return NextResponse.json({
      success: true,
      user: updatedProfile,
      message: 'Profile updated successfully'
    });
  } catch (error) {
    console.error(`[/api/social/users/${params.userId}] Error updating:`, error);
    return NextResponse.json(
      { error: error.message },
      { status: 500 }
    );
  }
}