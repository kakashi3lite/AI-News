import React, { useState } from 'react';
import { Button } from './ui/button';
import { Popover, PopoverContent, PopoverTrigger } from './ui/popover';
import { Textarea } from './ui/textarea';
import { motion } from 'framer-motion';
import { 
  Share2, 
  User, 
  Twitter, 
  Linkedin, 
  Facebook, 
  Copy, 
  MessageSquare,
  Check,
  ExternalLink
} from 'lucide-react';
import { toast } from 'react-hot-toast';

/**
 * ShareMenu Component
 * Provides sharing options for articles including profile sharing and external platforms
 * @param {Object} props
 * @param {Object} props.article - Article object with title, url, description
 * @param {Function} props.onShare - Callback when article is shared
 * @param {string} props.size - 'sm' | 'md' | 'lg'
 */
const ShareMenu = ({
  article,
  onShare,
  size = 'md'
}) => {
  const [isOpen, setIsOpen] = useState(false);
  const [shareNote, setShareNote] = useState('');
  const [loading, setLoading] = useState(false);
  const [copied, setCopied] = useState(false);

  const shareOptions = [
    {
      id: 'profile',
      label: 'Share to Profile',
      icon: User,
      color: 'text-blue-600',
      action: 'profile'
    },
    {
      id: 'twitter',
      label: 'Share on Twitter',
      icon: Twitter,
      color: 'text-sky-500',
      action: 'external'
    },
    {
      id: 'linkedin',
      label: 'Share on LinkedIn',
      icon: Linkedin,
      color: 'text-blue-700',
      action: 'external'
    },
    {
      id: 'facebook',
      label: 'Share on Facebook',
      icon: Facebook,
      color: 'text-blue-600',
      action: 'external'
    },
    {
      id: 'copy',
      label: 'Copy Link',
      icon: copied ? Check : Copy,
      color: copied ? 'text-green-600' : 'text-gray-600',
      action: 'copy'
    }
  ];

  const handleProfileShare = async () => {
    if (loading) return;
    
    setLoading(true);
    try {
      const response = await fetch('/api/social/share', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          articleId: article.id,
          articleUrl: article.url,
          articleTitle: article.title,
          note: shareNote.trim(),
          shareType: 'profile'
        })
      });

      if (response.ok) {
        toast.success('Article shared to your profile!');
        setShareNote('');
        setIsOpen(false);
        onShare?.('profile', { note: shareNote });
      } else {
        toast.error('Failed to share article');
      }
    } catch (error) {
      console.error('Error sharing article:', error);
      toast.error('Error sharing article');
    } finally {
      setLoading(false);
    }
  };

  const handleExternalShare = (platform) => {
    const url = encodeURIComponent(article.url);
    const title = encodeURIComponent(article.title);
    const description = encodeURIComponent(article.description || '');
    
    let shareUrl = '';
    
    switch (platform) {
      case 'twitter':
        shareUrl = `https://twitter.com/intent/tweet?url=${url}&text=${title}`;
        break;
      case 'linkedin':
        shareUrl = `https://www.linkedin.com/sharing/share-offsite/?url=${url}`;
        break;
      case 'facebook':
        shareUrl = `https://www.facebook.com/sharer/sharer.php?u=${url}`;
        break;
      default:
        return;
    }
    
    window.open(shareUrl, '_blank', 'width=600,height=400');
    onShare?.(platform, { url: article.url });
    setIsOpen(false);
  };

  const handleCopyLink = async () => {
    try {
      await navigator.clipboard.writeText(article.url);
      setCopied(true);
      toast.success('Link copied to clipboard!');
      setTimeout(() => setCopied(false), 2000);
      onShare?.('copy', { url: article.url });
    } catch (error) {
      console.error('Failed to copy link:', error);
      toast.error('Failed to copy link');
    }
  };

  const handleOptionClick = (option) => {
    switch (option.action) {
      case 'profile':
        // Keep popover open for profile sharing to allow note input
        break;
      case 'external':
        handleExternalShare(option.id);
        break;
      case 'copy':
        handleCopyLink();
        break;
    }
  };

  const sizeClasses = {
    sm: 'h-6 w-6',
    md: 'h-8 w-8',
    lg: 'h-10 w-10'
  };

  return (
    <Popover open={isOpen} onOpenChange={setIsOpen}>
      <PopoverTrigger asChild>
        <Button
          variant="ghost"
          size={size}
          className={`${sizeClasses[size]} hover:bg-gray-100 transition-colors`}
          title="Share article"
        >
          <Share2 className="h-4 w-4" />
        </Button>
      </PopoverTrigger>
      
      <PopoverContent className="w-80 p-4" align="end">
        <div className="space-y-4">
          <h3 className="font-semibold text-sm text-gray-900">Share Article</h3>
          
          {/* Profile Share Section */}
          <div className="space-y-3 p-3 bg-blue-50 rounded-lg">
            <div className="flex items-center gap-2 text-blue-700">
              <User className="h-4 w-4" />
              <span className="font-medium text-sm">Share to Your Profile</span>
            </div>
            
            <Textarea
              placeholder="Add a note about this article (optional)..."
              value={shareNote}
              onChange={(e) => setShareNote(e.target.value)}
              className="min-h-[60px] text-sm"
              maxLength={280}
            />
            
            <div className="flex justify-between items-center">
              <span className="text-xs text-gray-500">
                {shareNote.length}/280 characters
              </span>
              
              <Button
                onClick={handleProfileShare}
                disabled={loading}
                size="sm"
                className="bg-blue-600 hover:bg-blue-700"
              >
                {loading ? 'Sharing...' : 'Share'}
              </Button>
            </div>
          </div>
          
          {/* External Share Options */}
          <div className="space-y-2">
            <h4 className="text-xs font-medium text-gray-700 uppercase tracking-wide">
              External Platforms
            </h4>
            
            <div className="grid grid-cols-2 gap-2">
              {shareOptions.filter(option => option.action !== 'profile').map((option) => {
                const Icon = option.icon;
                return (
                  <motion.button
                    key={option.id}
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                    onClick={() => handleOptionClick(option)}
                    className="flex items-center gap-2 p-2 rounded-lg hover:bg-gray-100 transition-colors text-left"
                  >
                    <Icon className={`h-4 w-4 ${option.color}`} />
                    <span className="text-sm text-gray-700">{option.label}</span>
                    {option.action === 'external' && (
                      <ExternalLink className="h-3 w-3 text-gray-400 ml-auto" />
                    )}
                  </motion.button>
                );
              })}
            </div>
          </div>
        </div>
      </PopoverContent>
    </Popover>
  );
};

export default ShareMenu;