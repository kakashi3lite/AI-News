import React, { useState, useEffect } from 'react';
import { Button } from './ui/button';
import { Textarea } from './ui/textarea';
import { Avatar, AvatarFallback, AvatarImage } from './ui/avatar';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  MessageSquare, 
  ThumbsUp, 
  ThumbsDown, 
  Reply, 
  MoreHorizontal,
  Flag,
  Trash2,
  Edit3,
  Send
} from 'lucide-react';
import { Popover, PopoverContent, PopoverTrigger } from './ui/popover';
import { toast } from 'react-hot-toast';

/**
 * Individual Comment Component
 */
const Comment = ({ 
  comment, 
  onReply, 
  onVote, 
  onEdit, 
  onDelete, 
  onFlag, 
  depth = 0,
  currentUserId 
}) => {
  const [isReplying, setIsReplying] = useState(false);
  const [isEditing, setIsEditing] = useState(false);
  const [replyText, setReplyText] = useState('');
  const [editText, setEditText] = useState(comment.content);
  const [loading, setLoading] = useState(false);

  const handleReply = async () => {
    if (!replyText.trim() || loading) return;
    
    setLoading(true);
    try {
      await onReply(comment.id, replyText.trim());
      setReplyText('');
      setIsReplying(false);
    } catch (error) {
      toast.error('Failed to post reply');
    } finally {
      setLoading(false);
    }
  };

  const handleEdit = async () => {
    if (!editText.trim() || loading) return;
    
    setLoading(true);
    try {
      await onEdit(comment.id, editText.trim());
      setIsEditing(false);
    } catch (error) {
      toast.error('Failed to edit comment');
    } finally {
      setLoading(false);
    }
  };

  const isOwner = currentUserId === comment.userId;
  const maxDepth = 3;
  const canReply = depth < maxDepth;

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className={`${depth > 0 ? 'ml-8 border-l-2 border-gray-100 pl-4' : ''}`}
    >
      <div className="flex gap-3 py-3">
        <Avatar className="h-8 w-8 flex-shrink-0">
          <AvatarImage src={comment.user?.avatar} />
          <AvatarFallback>
            {comment.user?.name?.charAt(0) || 'U'}
          </AvatarFallback>
        </Avatar>
        
        <div className="flex-1 space-y-2">
          <div className="flex items-center gap-2">
            <span className="font-medium text-sm text-gray-900">
              {comment.user?.name || 'Anonymous'}
            </span>
            <span className="text-xs text-gray-500">
              {new Date(comment.createdAt).toLocaleDateString()}
            </span>
            {comment.isEdited && (
              <span className="text-xs text-gray-400">(edited)</span>
            )}
          </div>
          
          {isEditing ? (
            <div className="space-y-2">
              <Textarea
                value={editText}
                onChange={(e) => setEditText(e.target.value)}
                className="min-h-[60px] text-sm"
                placeholder="Edit your comment..."
              />
              <div className="flex gap-2">
                <Button
                  size="sm"
                  onClick={handleEdit}
                  disabled={loading || !editText.trim()}
                >
                  Save
                </Button>
                <Button
                  size="sm"
                  variant="outline"
                  onClick={() => {
                    setIsEditing(false);
                    setEditText(comment.content);
                  }}
                >
                  Cancel
                </Button>
              </div>
            </div>
          ) : (
            <p className="text-sm text-gray-700 whitespace-pre-wrap">
              {comment.content}
            </p>
          )}
          
          <div className="flex items-center gap-4">
            {/* Vote buttons */}
            <div className="flex items-center gap-1">
              <Button
                variant="ghost"
                size="sm"
                onClick={() => onVote(comment.id, 'up')}
                className={`h-6 px-2 ${
                  comment.userVote === 'up' ? 'text-green-600 bg-green-50' : 'text-gray-500'
                }`}
              >
                <ThumbsUp className="h-3 w-3" />
                <span className="ml-1 text-xs">{comment.upvotes || 0}</span>
              </Button>
              
              <Button
                variant="ghost"
                size="sm"
                onClick={() => onVote(comment.id, 'down')}
                className={`h-6 px-2 ${
                  comment.userVote === 'down' ? 'text-red-600 bg-red-50' : 'text-gray-500'
                }`}
              >
                <ThumbsDown className="h-3 w-3" />
                <span className="ml-1 text-xs">{comment.downvotes || 0}</span>
              </Button>
            </div>
            
            {/* Reply button */}
            {canReply && (
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setIsReplying(!isReplying)}
                className="h-6 px-2 text-gray-500 hover:text-blue-600"
              >
                <Reply className="h-3 w-3" />
                <span className="ml-1 text-xs">Reply</span>
              </Button>
            )}
            
            {/* More options */}
            <Popover>
              <PopoverTrigger asChild>
                <Button
                  variant="ghost"
                  size="sm"
                  className="h-6 w-6 p-0 text-gray-400 hover:text-gray-600"
                >
                  <MoreHorizontal className="h-3 w-3" />
                </Button>
              </PopoverTrigger>
              <PopoverContent className="w-40 p-1" align="end">
                <div className="space-y-1">
                  {isOwner && (
                    <>
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => setIsEditing(true)}
                        className="w-full justify-start h-8 px-2"
                      >
                        <Edit3 className="h-3 w-3 mr-2" />
                        Edit
                      </Button>
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => onDelete(comment.id)}
                        className="w-full justify-start h-8 px-2 text-red-600 hover:text-red-700"
                      >
                        <Trash2 className="h-3 w-3 mr-2" />
                        Delete
                      </Button>
                    </>
                  )}
                  {!isOwner && (
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => onFlag(comment.id)}
                      className="w-full justify-start h-8 px-2 text-orange-600 hover:text-orange-700"
                    >
                      <Flag className="h-3 w-3 mr-2" />
                      Report
                    </Button>
                  )}
                </div>
              </PopoverContent>
            </Popover>
          </div>
          
          {/* Reply form */}
          <AnimatePresence>
            {isReplying && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: 'auto' }}
                exit={{ opacity: 0, height: 0 }}
                className="space-y-2 pt-2"
              >
                <Textarea
                  value={replyText}
                  onChange={(e) => setReplyText(e.target.value)}
                  placeholder="Write a reply..."
                  className="min-h-[60px] text-sm"
                />
                <div className="flex gap-2">
                  <Button
                    size="sm"
                    onClick={handleReply}
                    disabled={loading || !replyText.trim()}
                  >
                    <Send className="h-3 w-3 mr-1" />
                    Reply
                  </Button>
                  <Button
                    size="sm"
                    variant="outline"
                    onClick={() => {
                      setIsReplying(false);
                      setReplyText('');
                    }}
                  >
                    Cancel
                  </Button>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>
      
      {/* Nested replies */}
      {comment.replies && comment.replies.length > 0 && (
        <div className="space-y-0">
          {comment.replies.map((reply) => (
            <Comment
              key={reply.id}
              comment={reply}
              onReply={onReply}
              onVote={onVote}
              onEdit={onEdit}
              onDelete={onDelete}
              onFlag={onFlag}
              depth={depth + 1}
              currentUserId={currentUserId}
            />
          ))}
        </div>
      )}
    </motion.div>
  );
};

/**
 * CommentThread Component
 * Main component for displaying and managing comments on articles
 */
const CommentThread = ({ articleId, currentUserId }) => {
  const [comments, setComments] = useState([]);
  const [newComment, setNewComment] = useState('');
  const [loading, setLoading] = useState(false);
  const [isExpanded, setIsExpanded] = useState(false);

  useEffect(() => {
    if (isExpanded) {
      fetchComments();
    }
  }, [isExpanded, articleId]);

  const fetchComments = async () => {
    try {
      const response = await fetch(`/api/social/comments?articleId=${articleId}`);
      if (response.ok) {
        const data = await response.json();
        setComments(data.comments || []);
      }
    } catch (error) {
      console.error('Error fetching comments:', error);
    }
  };

  const handleAddComment = async () => {
    if (!newComment.trim() || loading) return;
    
    setLoading(true);
    try {
      const response = await fetch('/api/social/comments', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          articleId,
          content: newComment.trim(),
          parentId: null
        })
      });

      if (response.ok) {
        const data = await response.json();
        setComments([data.comment, ...comments]);
        setNewComment('');
        toast.success('Comment posted!');
      } else {
        toast.error('Failed to post comment');
      }
    } catch (error) {
      console.error('Error posting comment:', error);
      toast.error('Error posting comment');
    } finally {
      setLoading(false);
    }
  };

  const handleReply = async (parentId, content) => {
    const response = await fetch('/api/social/comments', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        articleId,
        content,
        parentId
      })
    });

    if (response.ok) {
      await fetchComments(); // Refresh to show new reply
      toast.success('Reply posted!');
    } else {
      throw new Error('Failed to post reply');
    }
  };

  const handleVote = async (commentId, voteType) => {
    try {
      const response = await fetch('/api/social/comments/vote', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          commentId,
          voteType
        })
      });

      if (response.ok) {
        await fetchComments(); // Refresh to show updated votes
      }
    } catch (error) {
      console.error('Error voting on comment:', error);
    }
  };

  const handleEdit = async (commentId, content) => {
    const response = await fetch(`/api/social/comments/${commentId}`, {
      method: 'PUT',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ content })
    });

    if (response.ok) {
      await fetchComments();
      toast.success('Comment updated!');
    } else {
      throw new Error('Failed to update comment');
    }
  };

  const handleDelete = async (commentId) => {
    if (!confirm('Are you sure you want to delete this comment?')) return;
    
    try {
      const response = await fetch(`/api/social/comments/${commentId}`, {
        method: 'DELETE'
      });

      if (response.ok) {
        await fetchComments();
        toast.success('Comment deleted!');
      } else {
        toast.error('Failed to delete comment');
      }
    } catch (error) {
      console.error('Error deleting comment:', error);
      toast.error('Error deleting comment');
    }
  };

  const handleFlag = async (commentId) => {
    try {
      const response = await fetch('/api/social/comments/flag', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ commentId })
      });

      if (response.ok) {
        toast.success('Comment reported. Thank you for helping keep our community safe.');
      } else {
        toast.error('Failed to report comment');
      }
    } catch (error) {
      console.error('Error flagging comment:', error);
      toast.error('Error reporting comment');
    }
  };

  return (
    <div className="border-t border-gray-200 pt-4">
      <Button
        variant="ghost"
        onClick={() => setIsExpanded(!isExpanded)}
        className="flex items-center gap-2 text-gray-600 hover:text-gray-900 mb-4"
      >
        <MessageSquare className="h-4 w-4" />
        <span>{isExpanded ? 'Hide' : 'Show'} Comments ({comments.length})</span>
      </Button>
      
      <AnimatePresence>
        {isExpanded && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="space-y-4"
          >
            {/* New comment form */}
            <div className="space-y-3">
              <Textarea
                value={newComment}
                onChange={(e) => setNewComment(e.target.value)}
                placeholder="Share your thoughts on this article..."
                className="min-h-[80px]"
              />
              <div className="flex justify-end">
                <Button
                  onClick={handleAddComment}
                  disabled={loading || !newComment.trim()}
                >
                  <Send className="h-4 w-4 mr-2" />
                  {loading ? 'Posting...' : 'Post Comment'}
                </Button>
              </div>
            </div>
            
            {/* Comments list */}
            <div className="space-y-0 divide-y divide-gray-100">
              {comments.map((comment) => (
                <Comment
                  key={comment.id}
                  comment={comment}
                  onReply={handleReply}
                  onVote={handleVote}
                  onEdit={handleEdit}
                  onDelete={handleDelete}
                  onFlag={handleFlag}
                  currentUserId={currentUserId}
                />
              ))}
            </div>
            
            {comments.length === 0 && (
              <div className="text-center py-8 text-gray-500">
                <MessageSquare className="h-12 w-12 mx-auto mb-3 text-gray-300" />
                <p>No comments yet. Be the first to share your thoughts!</p>
              </div>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default CommentThread;