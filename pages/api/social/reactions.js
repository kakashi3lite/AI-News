// API endpoint for handling article reactions (emoji responses)
// Supports adding, removing, and fetching reactions

export default async function handler(req, res) {
  const { method } = req;
  const currentUserId = req.headers['user-id'] || 'demo-user-1';

  switch (method) {
    case 'GET':
      return handleGetReactions(req, res);
    case 'POST':
      return handleAddReaction(req, res, currentUserId);
    case 'DELETE':
      return handleRemoveReaction(req, res, currentUserId);
    default:
      return res.status(405).json({ error: 'Method not allowed' });
  }
}

// Get reactions for an article
async function handleGetReactions(req, res) {
  try {
    const { articleId } = req.query;

    if (!articleId) {
      return res.status(400).json({ error: 'Article ID is required' });
    }

    // Simulate fetching reactions from database
    // In a real app, this would query the reactions table
    const mockReactions = {
      articleId,
      reactions: {
        '👍': { count: 12, users: ['user1', 'user2'] },
        '❤️': { count: 8, users: ['user3', 'user4'] },
        '🔥': { count: 5, users: ['user5'] },
        '💡': { count: 3, users: [] },
        '🤔': { count: 1, users: ['user6'] }
      },
      totalReactions: 29,
      userReactions: [] // Reactions by current user
    };

    return res.status(200).json(mockReactions);
  } catch (error) {
    console.error('Error fetching reactions:', error);
    return res.status(500).json({ error: 'Internal server error' });
  }
}

// Add a reaction to an article
async function handleAddReaction(req, res, currentUserId) {
  try {
    const { articleId, emoji } = req.body;

    if (!articleId || !emoji) {
      return res.status(400).json({ error: 'Article ID and emoji are required' });
    }

    // Validate emoji (only allow specific reactions)
    const allowedEmojis = ['👍', '❤️', '🔥', '💡', '🤔'];
    if (!allowedEmojis.includes(emoji)) {
      return res.status(400).json({ error: 'Invalid emoji reaction' });
    }

    // Simulate database operations
    const reactionData = {
      userId: currentUserId,
      articleId,
      emoji,
      createdAt: new Date().toISOString()
    };

    console.log('Adding reaction:', reactionData);

    // In a real app, you would:
    // 1. Check if user already reacted with this emoji
    // 2. If yes, remove the existing reaction (toggle behavior)
    // 3. If no, add the new reaction
    // 4. Update reaction counts
    // 5. Create notification for article author

    // Simulate notification creation
    const notification = {
      id: `notif-${Date.now()}`,
      type: 'reaction',
      actorId: currentUserId,
      targetId: articleId,
      emoji: emoji,
      message: `Someone reacted to your shared article`,
      isRead: false,
      createdAt: new Date().toISOString()
    };
    console.log('Created notification:', notification);

    return res.status(200).json({
      success: true,
      action: 'added',
      reaction: reactionData,
      message: 'Reaction added successfully'
    });

  } catch (error) {
    console.error('Error adding reaction:', error);
    return res.status(500).json({ error: 'Internal server error' });
  }
}

// Remove a reaction from an article
async function handleRemoveReaction(req, res, currentUserId) {
  try {
    const { articleId, emoji } = req.body;

    if (!articleId || !emoji) {
      return res.status(400).json({ error: 'Article ID and emoji are required' });
    }

    // Simulate removing reaction
    console.log('Removing reaction:', { userId: currentUserId, articleId, emoji });

    // In a real app, you would:
    // 1. Delete the reaction record
    // 2. Update reaction counts

    return res.status(200).json({
      success: true,
      action: 'removed',
      message: 'Reaction removed successfully'
    });

  } catch (error) {
    console.error('Error removing reaction:', error);
    return res.status(500).json({ error: 'Internal server error' });
  }
}

// Example database schema for reactions table:
/*
CREATE TABLE reactions (
  id SERIAL PRIMARY KEY,
  user_id VARCHAR(255) NOT NULL,
  article_id VARCHAR(255) NOT NULL,
  emoji VARCHAR(10) NOT NULL,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  UNIQUE(user_id, article_id, emoji)
);

CREATE INDEX idx_reactions_article_id ON reactions(article_id);
CREATE INDEX idx_reactions_user_id ON reactions(user_id);
*/