// API endpoint for handling article sharing and resharing
// Supports sharing to profile with notes and external platforms

export default async function handler(req, res) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  try {
    const { articleId, shareType, note = '', platform } = req.body;
    const currentUserId = req.headers['user-id'] || 'demo-user-1';

    if (!articleId || !shareType) {
      return res.status(400).json({ error: 'Article ID and share type are required' });
    }

    if (!['profile', 'external'].includes(shareType)) {
      return res.status(400).json({ error: 'Invalid share type' });
    }

    if (shareType === 'profile') {
      return handleProfileShare(req, res, currentUserId, articleId, note);
    } else {
      return handleExternalShare(req, res, articleId, platform);
    }

  } catch (error) {
    console.error('Error in share API:', error);
    return res.status(500).json({ error: 'Internal server error' });
  }
}

// Handle sharing to user's profile
async function handleProfileShare(req, res, currentUserId, articleId, note) {
  try {
    // Validate note length
    if (note && note.length > 500) {
      return res.status(400).json({ error: 'Note must be less than 500 characters' });
    }

    // Create share record
    const shareData = {
      id: `share-${Date.now()}`,
      userId: currentUserId,
      articleId,
      note: note.trim(),
      shareType: 'profile',
      createdAt: new Date().toISOString(),
      isPublic: true,
      engagement: {
        views: 0,
        clicks: 0,
        reactions: 0,
        comments: 0
      }
    };

    console.log('Creating profile share:', shareData);

    // In a real app, you would:
    // 1. Insert share record into database
    // 2. Update user's activity feed
    // 3. Update article share count
    // 4. Create notifications for followers
    // 5. Index for search and recommendations

    // Simulate follower notifications
    const followerNotification = {
      id: `notif-${Date.now()}`,
      type: 'share',
      actorId: currentUserId,
      targetId: articleId,
      message: 'Someone you follow shared an article',
      shareNote: note,
      isRead: false,
      createdAt: new Date().toISOString()
    };
    console.log('Created follower notification:', followerNotification);

    return res.status(201).json({
      success: true,
      share: shareData,
      message: 'Article shared to your profile successfully'
    });

  } catch (error) {
    console.error('Error in profile share:', error);
    return res.status(500).json({ error: 'Internal server error' });
  }
}

// Handle external platform sharing
async function handleExternalShare(req, res, articleId, platform) {
  try {
    const validPlatforms = ['twitter', 'linkedin', 'facebook', 'copy'];
    
    if (!platform || !validPlatforms.includes(platform)) {
      return res.status(400).json({ error: 'Invalid or missing platform' });
    }

    // Mock article data (in real app, fetch from database)
    const article = {
      id: articleId,
      title: 'Sample Article Title',
      description: 'This is a sample article description that provides insights into the topic.',
      url: `https://example.com/articles/${articleId}`,
      image: 'https://example.com/images/article-thumbnail.jpg',
      author: 'John Doe',
      publishedAt: '2024-01-15T10:00:00Z'
    };

    // Generate platform-specific share URLs
    const shareUrls = {
      twitter: generateTwitterUrl(article),
      linkedin: generateLinkedInUrl(article),
      facebook: generateFacebookUrl(article),
      copy: article.url
    };

    // Log external share for analytics
    const externalShareData = {
      userId: req.headers['user-id'] || 'demo-user-1',
      articleId,
      platform,
      shareUrl: shareUrls[platform],
      createdAt: new Date().toISOString()
    };

    console.log('External share:', externalShareData);

    // In a real app, you would:
    // 1. Log the external share for analytics
    // 2. Update article external share count
    // 3. Track user engagement patterns

    return res.status(200).json({
      success: true,
      platform,
      shareUrl: shareUrls[platform],
      message: `Share URL generated for ${platform}`
    });

  } catch (error) {
    console.error('Error in external share:', error);
    return res.status(500).json({ error: 'Internal server error' });
  }
}

// Helper functions to generate platform-specific share URLs
function generateTwitterUrl(article) {
  const text = encodeURIComponent(`${article.title} by ${article.author}`);
  const url = encodeURIComponent(article.url);
  return `https://twitter.com/intent/tweet?text=${text}&url=${url}`;
}

function generateLinkedInUrl(article) {
  const url = encodeURIComponent(article.url);
  const title = encodeURIComponent(article.title);
  const summary = encodeURIComponent(article.description);
  return `https://www.linkedin.com/sharing/share-offsite/?url=${url}&title=${title}&summary=${summary}`;
}

function generateFacebookUrl(article) {
  const url = encodeURIComponent(article.url);
  return `https://www.facebook.com/sharer/sharer.php?u=${url}`;
}

// Example database schema for shares table:
/*
CREATE TABLE shares (
  id SERIAL PRIMARY KEY,
  user_id VARCHAR(255) NOT NULL,
  article_id VARCHAR(255) NOT NULL,
  share_type VARCHAR(50) NOT NULL, -- 'profile', 'external'
  platform VARCHAR(50), -- for external shares
  note TEXT,
  is_public BOOLEAN DEFAULT TRUE,
  views INTEGER DEFAULT 0,
  clicks INTEGER DEFAULT 0,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE external_shares (
  id SERIAL PRIMARY KEY,
  user_id VARCHAR(255) NOT NULL,
  article_id VARCHAR(255) NOT NULL,
  platform VARCHAR(50) NOT NULL,
  share_url TEXT,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_shares_user_id ON shares(user_id);
CREATE INDEX idx_shares_article_id ON shares(article_id);
CREATE INDEX idx_external_shares_platform ON external_shares(platform);
*/