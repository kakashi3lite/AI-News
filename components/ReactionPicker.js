import React, { useState, useRef, useEffect } from 'react';
import { Button } from './ui/button';
import { Popover, PopoverContent, PopoverTrigger } from './ui/popover';
import { motion, AnimatePresence } from 'framer-motion';
import { Heart, ThumbsUp, Laugh, Angry, Sad } from 'lucide-react';

/**
 * ReactionPicker Component
 * Provides emoji reactions for articles with 5 different emotions
 * @param {Object} props
 * @param {string} props.articleId - ID of the article
 * @param {Object} props.reactions - Current reaction counts {like: 5, love: 2, ...}
 * @param {string} props.userReaction - Current user's reaction
 * @param {Function} props.onReactionChange - Callback when reaction changes
 * @param {string} props.size - 'sm' | 'md' | 'lg'
 */
const ReactionPicker = ({
  articleId,
  reactions = {},
  userReaction = null,
  onReactionChange,
  size = 'md'
}) => {
  const [isOpen, setIsOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const [localReactions, setLocalReactions] = useState(reactions);
  const [localUserReaction, setLocalUserReaction] = useState(userReaction);

  const reactionTypes = [
    { 
      id: 'like', 
      emoji: '👍', 
      icon: ThumbsUp, 
      label: 'Like',
      color: 'text-blue-500'
    },
    { 
      id: 'love', 
      emoji: '❤️', 
      icon: Heart, 
      label: 'Love',
      color: 'text-red-500'
    },
    { 
      id: 'laugh', 
      emoji: '😂', 
      icon: Laugh, 
      label: 'Funny',
      color: 'text-yellow-500'
    },
    { 
      id: 'angry', 
      emoji: '😠', 
      icon: Angry, 
      label: 'Angry',
      color: 'text-red-600'
    },
    { 
      id: 'sad', 
      emoji: '😢', 
      icon: Sad, 
      label: 'Sad',
      color: 'text-blue-600'
    }
  ];

  const handleReaction = async (reactionType) => {
    if (loading) return;
    
    setLoading(true);
    setIsOpen(false);
    
    try {
      const isRemoving = localUserReaction === reactionType;
      const response = await fetch('/api/social/reactions', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          articleId,
          reactionType: isRemoving ? null : reactionType,
          action: isRemoving ? 'remove' : 'add'
        })
      });

      if (response.ok) {
        const data = await response.json();
        
        // Update local state optimistically
        const newReactions = { ...localReactions };
        
        // Remove previous reaction
        if (localUserReaction && newReactions[localUserReaction]) {
          newReactions[localUserReaction] = Math.max(0, newReactions[localUserReaction] - 1);
        }
        
        // Add new reaction
        if (!isRemoving) {
          newReactions[reactionType] = (newReactions[reactionType] || 0) + 1;
        }
        
        setLocalReactions(newReactions);
        setLocalUserReaction(isRemoving ? null : reactionType);
        onReactionChange?.(newReactions, isRemoving ? null : reactionType);
      }
    } catch (error) {
      console.error('Error updating reaction:', error);
    } finally {
      setLoading(false);
    }
  };

  const getTotalReactions = () => {
    return Object.values(localReactions).reduce((sum, count) => sum + count, 0);
  };

  const getCurrentReactionEmoji = () => {
    if (!localUserReaction) return '👍';
    const reaction = reactionTypes.find(r => r.id === localUserReaction);
    return reaction?.emoji || '👍';
  };

  const sizeClasses = {
    sm: 'h-6 w-6 text-xs',
    md: 'h-8 w-8 text-sm',
    lg: 'h-10 w-10 text-base'
  };

  return (
    <div className="flex items-center gap-2">
      <Popover open={isOpen} onOpenChange={setIsOpen}>
        <PopoverTrigger asChild>
          <Button
            variant="ghost"
            size={size}
            className={`
              ${sizeClasses[size]} 
              ${localUserReaction ? 'bg-blue-50 text-blue-600 hover:bg-blue-100' : 'hover:bg-gray-100'}
              transition-all duration-200
            `}
            disabled={loading}
          >
            <span className="text-lg">{getCurrentReactionEmoji()}</span>
          </Button>
        </PopoverTrigger>
        
        <PopoverContent className="w-auto p-2" align="start">
          <div className="flex gap-1">
            {reactionTypes.map((reaction) => (
              <motion.button
                key={reaction.id}
                whileHover={{ scale: 1.2 }}
                whileTap={{ scale: 0.9 }}
                onClick={() => handleReaction(reaction.id)}
                className={`
                  p-2 rounded-lg hover:bg-gray-100 transition-colors
                  ${localUserReaction === reaction.id ? 'bg-blue-50 ring-2 ring-blue-200' : ''}
                `}
                title={reaction.label}
              >
                <span className="text-xl">{reaction.emoji}</span>
              </motion.button>
            ))}
          </div>
        </PopoverContent>
      </Popover>
      
      {getTotalReactions() > 0 && (
        <motion.span 
          initial={{ opacity: 0, scale: 0.8 }}
          animate={{ opacity: 1, scale: 1 }}
          className="text-sm text-gray-600 font-medium"
        >
          {getTotalReactions()}
        </motion.span>
      )}
      
      {/* Show top reactions */}
      {getTotalReactions() > 0 && (
        <div className="flex gap-1">
          {reactionTypes
            .filter(reaction => localReactions[reaction.id] > 0)
            .sort((a, b) => (localReactions[b.id] || 0) - (localReactions[a.id] || 0))
            .slice(0, 3)
            .map(reaction => (
              <motion.div
                key={reaction.id}
                initial={{ opacity: 0, scale: 0 }}
                animate={{ opacity: 1, scale: 1 }}
                className="flex items-center gap-1 text-xs text-gray-500"
              >
                <span>{reaction.emoji}</span>
                <span>{localReactions[reaction.id]}</span>
              </motion.div>
            ))
          }
        </div>
      )}
    </div>
  );
};

export default ReactionPicker;