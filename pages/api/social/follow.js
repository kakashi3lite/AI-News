// API endpoint for handling follow/unfollow actions
// Supports following topics, authors, and users

export default async function handler(req, res) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  try {
    const { type, targetId, action } = req.body;

    // Validate input
    if (!type || !targetId || !action) {
      return res.status(400).json({ error: 'Missing required fields' });
    }

    if (!['topic', 'author', 'user'].includes(type)) {
      return res.status(400).json({ error: 'Invalid type' });
    }

    if (!['follow', 'unfollow'].includes(action)) {
      return res.status(400).json({ error: 'Invalid action' });
    }

    // Get current user (in a real app, this would come from authentication)
    const currentUserId = req.headers['user-id'] || 'demo-user-1';

    // Simulate database operations
    const followData = {
      userId: currentUserId,
      targetType: type,
      targetId: targetId,
      createdAt: new Date().toISOString()
    };

    if (action === 'follow') {
      // Add follow relationship
      console.log('Adding follow:', followData);
      
      // In a real app, you would:
      // 1. Check if already following
      // 2. Insert into follows table
      // 3. Update follower/following counts
      // 4. Create notification for the followed entity (if user)
      
      // Simulate notification creation for user follows
      if (type === 'user') {
        const notification = {
          id: `notif-${Date.now()}`,
          type: 'follow',
          actorId: currentUserId,
          targetId: targetId,
          message: `You have a new follower`,
          isRead: false,
          createdAt: new Date().toISOString()
        };
        console.log('Created notification:', notification);
      }
      
      return res.status(200).json({ 
        success: true, 
        action: 'followed',
        message: `Successfully followed ${type}` 
      });
    } else {
      // Remove follow relationship
      console.log('Removing follow:', followData);
      
      // In a real app, you would:
      // 1. Delete from follows table
      // 2. Update follower/following counts
      
      return res.status(200).json({ 
        success: true, 
        action: 'unfollowed',
        message: `Successfully unfollowed ${type}` 
      });
    }

  } catch (error) {
    console.error('Error in follow API:', error);
    return res.status(500).json({ error: 'Internal server error' });
  }
}

// Example database schema for follows table:
/*
CREATE TABLE follows (
  id SERIAL PRIMARY KEY,
  user_id VARCHAR(255) NOT NULL,
  target_type VARCHAR(50) NOT NULL, -- 'topic', 'author', 'user'
  target_id VARCHAR(255) NOT NULL,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  UNIQUE(user_id, target_type, target_id)
);

CREATE INDEX idx_follows_user_id ON follows(user_id);
CREATE INDEX idx_follows_target ON follows(target_type, target_id);
*/