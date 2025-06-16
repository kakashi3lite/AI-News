// API endpoint for handling user profiles and activity streams
// Supports fetching profile data, updating profiles, and activity feeds

export default async function handler(req, res) {
  const { method } = req;
  const currentUserId = req.headers['user-id'] || 'demo-user-1';

  switch (method) {
    case 'GET':
      return handleGetProfile(req, res, currentUserId);
    case 'PUT':
      return handleUpdateProfile(req, res, currentUserId);
    default:
      return res.status(405).json({ error: 'Method not allowed' });
  }
}

// Get user profile and activity data
async function handleGetProfile(req, res, currentUserId) {
  try {
    const { userId = currentUserId, tab = 'activity' } = req.query;

    // Mock user profile data
    const userProfile = {
      id: userId,
      username: userId === currentUserId ? 'currentuser' : 'otheruser',
      displayName: userId === currentUserId ? 'Current User' : 'Other User',
      email: userId === currentUserId ? 'current@example.com' : 'other@example.com',
      avatar: '/avatars/default.jpg',
      bio: 'Passionate about technology, AI, and staying informed about the latest news trends.',
      website: 'https://example.com',
      location: 'San Francisco, CA',
      joinedAt: '2023-06-15T10:00:00Z',
      isVerified: false,
      interests: ['Technology', 'AI/ML', 'Startups', 'Climate Change', 'Space'],
      stats: {
        followers: 156,
        following: 89,
        sharedArticles: 42,
        reputation: 1250,
        totalReactions: 89,
        totalComments: 156
      },
      settings: {
        isPublic: true,
        showEmail: false,
        showActivity: true,
        allowFollows: true
      }
    };

    // Get activity data based on tab
    let activityData = {};
    
    switch (tab) {
      case 'activity':
        activityData = await getRecentActivity(userId);
        break;
      case 'shares':
        activityData = await getSharedArticles(userId);
        break;
      case 'topics':
        activityData = await getFollowedTopics(userId);
        break;
      default:
        activityData = await getRecentActivity(userId);
    }

    return res.status(200).json({
      profile: userProfile,
      activity: activityData,
      isOwnProfile: userId === currentUserId
    });

  } catch (error) {
    console.error('Error fetching profile:', error);
    return res.status(500).json({ error: 'Internal server error' });
  }
}

// Update user profile
async function handleUpdateProfile(req, res, currentUserId) {
  try {
    const {
      displayName,
      bio,
      website,
      location,
      interests,
      settings
    } = req.body;

    // Validate input
    if (displayName && displayName.length > 100) {
      return res.status(400).json({ error: 'Display name must be less than 100 characters' });
    }

    if (bio && bio.length > 500) {
      return res.status(400).json({ error: 'Bio must be less than 500 characters' });
    }

    if (website && !isValidUrl(website)) {
      return res.status(400).json({ error: 'Invalid website URL' });
    }

    if (interests && interests.length > 10) {
      return res.status(400).json({ error: 'Maximum 10 interests allowed' });
    }

    // Update profile data
    const updatedProfile = {
      id: currentUserId,
      displayName: displayName || 'Current User',
      bio: bio || '',
      website: website || '',
      location: location || '',
      interests: interests || [],
      settings: settings || {},
      updatedAt: new Date().toISOString()
    };

    console.log('Updating profile:', updatedProfile);

    // In a real app, you would:
    // 1. Validate user permissions
    // 2. Update user record in database
    // 3. Update search index
    // 4. Invalidate cache

    return res.status(200).json({
      success: true,
      profile: updatedProfile,
      message: 'Profile updated successfully'
    });

  } catch (error) {
    console.error('Error updating profile:', error);
    return res.status(500).json({ error: 'Internal server error' });
  }
}

// Helper function to get recent activity
async function getRecentActivity(userId) {
  // Mock recent activity data
  return {
    activities: [
      {
        id: 'activity-1',
        type: 'share',
        timestamp: '2024-01-15T14:30:00Z',
        article: {
          id: 'article-1',
          title: 'The Future of AI in Healthcare',
          url: 'https://example.com/ai-healthcare'
        },
        note: 'This is a fascinating look at how AI is transforming medical diagnosis.',
        engagement: { reactions: 5, comments: 2 }
      },
      {
        id: 'activity-2',
        type: 'comment',
        timestamp: '2024-01-15T12:15:00Z',
        article: {
          id: 'article-2',
          title: 'Climate Change Solutions for 2024',
          url: 'https://example.com/climate-solutions'
        },
        comment: 'Great insights on renewable energy adoption!',
        engagement: { upvotes: 8 }
      },
      {
        id: 'activity-3',
        type: 'reaction',
        timestamp: '2024-01-15T10:45:00Z',
        article: {
          id: 'article-3',
          title: 'SpaceX Mars Mission Update',
          url: 'https://example.com/spacex-mars'
        },
        emoji: '🚀'
      }
    ],
    totalCount: 25,
    hasMore: true
  };
}

// Helper function to get shared articles
async function getSharedArticles(userId) {
  return {
    shares: [
      {
        id: 'share-1',
        articleId: 'article-1',
        title: 'The Future of AI in Healthcare',
        description: 'Exploring how artificial intelligence is revolutionizing medical diagnosis and treatment.',
        url: 'https://example.com/ai-healthcare',
        image: 'https://example.com/images/ai-healthcare.jpg',
        note: 'This is a fascinating look at how AI is transforming medical diagnosis.',
        sharedAt: '2024-01-15T14:30:00Z',
        engagement: {
          views: 45,
          clicks: 12,
          reactions: 5,
          comments: 2
        }
      }
    ],
    totalCount: 42,
    hasMore: true
  };
}

// Helper function to get followed topics
async function getFollowedTopics(userId) {
  return {
    topics: [
      {
        id: 'topic-1',
        name: 'Artificial Intelligence',
        description: 'Latest developments in AI and machine learning',
        followerCount: 15420,
        articleCount: 1250,
        followedAt: '2023-08-15T10:00:00Z'
      },
      {
        id: 'topic-2',
        name: 'Climate Change',
        description: 'Environmental news and climate solutions',
        followerCount: 8930,
        articleCount: 890,
        followedAt: '2023-09-20T15:30:00Z'
      }
    ],
    totalCount: 12,
    hasMore: false
  };
}

// Helper function to validate URLs
function isValidUrl(string) {
  try {
    new URL(string);
    return true;
  } catch (_) {
    return false;
  }
}

// Example database schema for user profiles:
/*
CREATE TABLE user_profiles (
  id SERIAL PRIMARY KEY,
  user_id VARCHAR(255) UNIQUE NOT NULL,
  username VARCHAR(100) UNIQUE NOT NULL,
  display_name VARCHAR(100),
  email VARCHAR(255) UNIQUE NOT NULL,
  avatar_url TEXT,
  bio TEXT,
  website TEXT,
  location VARCHAR(255),
  interests JSONB,
  settings JSONB,
  is_verified BOOLEAN DEFAULT FALSE,
  is_public BOOLEAN DEFAULT TRUE,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE user_stats (
  user_id VARCHAR(255) PRIMARY KEY,
  followers_count INTEGER DEFAULT 0,
  following_count INTEGER DEFAULT 0,
  shared_articles_count INTEGER DEFAULT 0,
  reputation_score INTEGER DEFAULT 0,
  total_reactions INTEGER DEFAULT 0,
  total_comments INTEGER DEFAULT 0,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_user_profiles_username ON user_profiles(username);
CREATE INDEX idx_user_profiles_email ON user_profiles(email);
*/