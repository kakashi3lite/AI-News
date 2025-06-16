// API endpoint for handling comments and replies on articles
// Supports CRUD operations, nested replies, and voting

export default async function handler(req, res) {
  const { method } = req;
  const currentUserId = req.headers['user-id'] || 'demo-user-1';

  switch (method) {
    case 'GET':
      return handleGetComments(req, res);
    case 'POST':
      return handleCreateComment(req, res, currentUserId);
    case 'PUT':
      return handleUpdateComment(req, res, currentUserId);
    case 'DELETE':
      return handleDeleteComment(req, res, currentUserId);
    default:
      return res.status(405).json({ error: 'Method not allowed' });
  }
}

// Get comments for an article
async function handleGetComments(req, res) {
  try {
    const { articleId, sort = 'newest' } = req.query;

    if (!articleId) {
      return res.status(400).json({ error: 'Article ID is required' });
    }

    // Simulate fetching comments from database
    const mockComments = [
      {
        id: 'comment-1',
        articleId,
        userId: 'user-1',
        userName: 'Alice Johnson',
        userAvatar: '/avatars/alice.jpg',
        content: 'This is a really insightful article! Thanks for sharing.',
        createdAt: '2024-01-15T10:30:00Z',
        updatedAt: '2024-01-15T10:30:00Z',
        upvotes: 5,
        downvotes: 0,
        isEdited: false,
        isFlagged: false,
        parentId: null,
        replies: [
          {
            id: 'comment-2',
            articleId,
            userId: 'user-2',
            userName: 'Bob Smith',
            userAvatar: '/avatars/bob.jpg',
            content: 'I agree! The data visualization section was particularly well done.',
            createdAt: '2024-01-15T11:15:00Z',
            updatedAt: '2024-01-15T11:15:00Z',
            upvotes: 2,
            downvotes: 0,
            isEdited: false,
            isFlagged: false,
            parentId: 'comment-1',
            replies: []
          }
        ]
      },
      {
        id: 'comment-3',
        articleId,
        userId: 'user-3',
        userName: 'Carol Davis',
        userAvatar: '/avatars/carol.jpg',
        content: 'Could you provide more sources for the claims made in section 3?',
        createdAt: '2024-01-15T12:00:00Z',
        updatedAt: '2024-01-15T12:00:00Z',
        upvotes: 1,
        downvotes: 0,
        isEdited: false,
        isFlagged: false,
        parentId: null,
        replies: []
      }
    ];

    // Sort comments based on the sort parameter
    const sortedComments = sortComments(mockComments, sort);

    return res.status(200).json({
      comments: sortedComments,
      totalCount: mockComments.length,
      sort
    });

  } catch (error) {
    console.error('Error fetching comments:', error);
    return res.status(500).json({ error: 'Internal server error' });
  }
}

// Create a new comment or reply
async function handleCreateComment(req, res, currentUserId) {
  try {
    const { articleId, content, parentId = null } = req.body;

    if (!articleId || !content) {
      return res.status(400).json({ error: 'Article ID and content are required' });
    }

    if (content.trim().length < 3) {
      return res.status(400).json({ error: 'Comment must be at least 3 characters long' });
    }

    if (content.length > 2000) {
      return res.status(400).json({ error: 'Comment must be less than 2000 characters' });
    }

    // Create new comment
    const newComment = {
      id: `comment-${Date.now()}`,
      articleId,
      userId: currentUserId,
      userName: 'Current User', // In real app, fetch from user profile
      userAvatar: '/avatars/default.jpg',
      content: content.trim(),
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
      upvotes: 0,
      downvotes: 0,
      isEdited: false,
      isFlagged: false,
      parentId,
      replies: []
    };

    console.log('Creating comment:', newComment);

    // In a real app, you would:
    // 1. Insert comment into database
    // 2. Update comment count for article
    // 3. Create notification for article author (if not replying to own comment)
    // 4. Create notification for parent comment author (if reply)

    // Simulate notification creation
    if (!parentId) {
      // New top-level comment
      const notification = {
        id: `notif-${Date.now()}`,
        type: 'comment',
        actorId: currentUserId,
        targetId: articleId,
        message: 'Someone commented on your shared article',
        isRead: false,
        createdAt: new Date().toISOString()
      };
      console.log('Created notification:', notification);
    } else {
      // Reply to existing comment
      const notification = {
        id: `notif-${Date.now()}`,
        type: 'reply',
        actorId: currentUserId,
        targetId: parentId,
        message: 'Someone replied to your comment',
        isRead: false,
        createdAt: new Date().toISOString()
      };
      console.log('Created notification:', notification);
    }

    return res.status(201).json({
      success: true,
      comment: newComment,
      message: parentId ? 'Reply added successfully' : 'Comment added successfully'
    });

  } catch (error) {
    console.error('Error creating comment:', error);
    return res.status(500).json({ error: 'Internal server error' });
  }
}

// Update an existing comment
async function handleUpdateComment(req, res, currentUserId) {
  try {
    const { commentId, content } = req.body;

    if (!commentId || !content) {
      return res.status(400).json({ error: 'Comment ID and content are required' });
    }

    if (content.trim().length < 3) {
      return res.status(400).json({ error: 'Comment must be at least 3 characters long' });
    }

    // In a real app, you would:
    // 1. Verify user owns the comment
    // 2. Update the comment in database
    // 3. Set isEdited flag to true
    // 4. Update updatedAt timestamp

    console.log('Updating comment:', { commentId, content, userId: currentUserId });

    return res.status(200).json({
      success: true,
      message: 'Comment updated successfully'
    });

  } catch (error) {
    console.error('Error updating comment:', error);
    return res.status(500).json({ error: 'Internal server error' });
  }
}

// Delete a comment
async function handleDeleteComment(req, res, currentUserId) {
  try {
    const { commentId } = req.body;

    if (!commentId) {
      return res.status(400).json({ error: 'Comment ID is required' });
    }

    // In a real app, you would:
    // 1. Verify user owns the comment OR is admin
    // 2. Soft delete the comment (mark as deleted)
    // 3. Update comment count for article
    // 4. Handle nested replies (cascade or orphan)

    console.log('Deleting comment:', { commentId, userId: currentUserId });

    return res.status(200).json({
      success: true,
      message: 'Comment deleted successfully'
    });

  } catch (error) {
    console.error('Error deleting comment:', error);
    return res.status(500).json({ error: 'Internal server error' });
  }
}

// Helper function to sort comments
function sortComments(comments, sort) {
  switch (sort) {
    case 'oldest':
      return comments.sort((a, b) => new Date(a.createdAt) - new Date(b.createdAt));
    case 'newest':
      return comments.sort((a, b) => new Date(b.createdAt) - new Date(a.createdAt));
    case 'popular':
      return comments.sort((a, b) => (b.upvotes - b.downvotes) - (a.upvotes - a.downvotes));
    default:
      return comments;
  }
}

// Example database schema for comments table:
/*
CREATE TABLE comments (
  id SERIAL PRIMARY KEY,
  article_id VARCHAR(255) NOT NULL,
  user_id VARCHAR(255) NOT NULL,
  parent_id INTEGER REFERENCES comments(id),
  content TEXT NOT NULL,
  upvotes INTEGER DEFAULT 0,
  downvotes INTEGER DEFAULT 0,
  is_edited BOOLEAN DEFAULT FALSE,
  is_flagged BOOLEAN DEFAULT FALSE,
  is_deleted BOOLEAN DEFAULT FALSE,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_comments_article_id ON comments(article_id);
CREATE INDEX idx_comments_user_id ON comments(user_id);
CREATE INDEX idx_comments_parent_id ON comments(parent_id);
*/