import React, { useState } from 'react';
import { Button } from './ui/button';
import { UserPlus, UserMinus, Loader2 } from 'lucide-react';
import { motion } from 'framer-motion';

/**
 * FollowButton Component
 * Handles following/unfollowing for topics, authors, and users
 * @param {Object} props
 * @param {string} props.type - 'topic' | 'author' | 'user'
 * @param {string} props.targetId - ID of the entity to follow
 * @param {string} props.targetName - Display name of the entity
 * @param {boolean} props.isFollowing - Current follow state
 * @param {Function} props.onFollowChange - Callback when follow state changes
 * @param {string} props.size - 'sm' | 'md' | 'lg'
 * @param {string} props.variant - 'default' | 'outline' | 'ghost'
 */
const FollowButton = ({
  type = 'topic',
  targetId,
  targetName,
  isFollowing = false,
  onFollowChange,
  size = 'sm',
  variant = 'outline',
  className = ''
}) => {
  const [loading, setLoading] = useState(false);
  const [followState, setFollowState] = useState(isFollowing);

  const handleFollow = async () => {
    if (loading) return;
    
    setLoading(true);
    try {
      const response = await fetch('/api/social/follow', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          type,
          targetId,
          action: followState ? 'unfollow' : 'follow'
        })
      });

      if (response.ok) {
        const newState = !followState;
        setFollowState(newState);
        onFollowChange?.(newState, targetId, type);
      } else {
        console.error('Failed to update follow status');
      }
    } catch (error) {
      console.error('Error updating follow status:', error);
    } finally {
      setLoading(false);
    }
  };

  const getButtonText = () => {
    if (loading) return '';
    if (followState) {
      return type === 'topic' ? 'Following' : 'Following';
    }
    return type === 'topic' ? 'Follow Topic' : type === 'author' ? 'Follow Author' : 'Follow';
  };

  const getIcon = () => {
    if (loading) return <Loader2 className="h-4 w-4 animate-spin" />;
    return followState ? <UserMinus className="h-4 w-4" /> : <UserPlus className="h-4 w-4" />;
  };

  return (
    <motion.div
      whileHover={{ scale: 1.02 }}
      whileTap={{ scale: 0.98 }}
    >
      <Button
        variant={followState ? 'default' : variant}
        size={size}
        onClick={handleFollow}
        disabled={loading}
        className={`
          ${followState 
            ? 'bg-green-600 hover:bg-red-600 text-white' 
            : 'hover:bg-blue-600 hover:text-white'
          }
          transition-all duration-200 gap-2 ${className}
        `}
        title={`${followState ? 'Unfollow' : 'Follow'} ${targetName}`}
      >
        {getIcon()}
        <span className="hidden sm:inline">{getButtonText()}</span>
      </Button>
    </motion.div>
  );
};

export default FollowButton;