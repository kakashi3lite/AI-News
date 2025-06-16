import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  HeartIcon,
  UserGroupIcon,
  SparklesIcon,
  ChevronLeftIcon,
  ChevronRightIcon,
  EyeIcon,
  ShareIcon,
  ChatBubbleLeftIcon
} from '@heroicons/react/24/outline';
import { HeartIcon as HeartSolid } from '@heroicons/react/24/solid';

const SocialRecommendations = ({ 
  userId = 'demo-user-1', 
  type = 'network', 
  limit = 5,
  className = '' 
}) => {
  const [recommendations, setRecommendations] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [metadata, setMetadata] = useState(null);

  useEffect(() => {
    fetchRecommendations();
  }, [userId, type, limit]);

  const fetchRecommendations = async () => {
    try {
      setLoading(true);
      setError(null);
      
      const response = await fetch(
        `/api/social/recommendations?type=${type}&limit=${limit}`,
        {
          headers: {
            'user-id': userId
          }
        }
      );
      
      if (!response.ok) {
        throw new Error('Failed to fetch recommendations');
      }
      
      const data = await response.json();
      setRecommendations(data.recommendations || []);
      setMetadata(data.metadata);
    } catch (err) {
      console.error('Error fetching recommendations:', err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const nextRecommendation = () => {
    setCurrentIndex((prev) => 
      prev === recommendations.length - 1 ? 0 : prev + 1
    );
  };

  const prevRecommendation = () => {
    setCurrentIndex((prev) => 
      prev === 0 ? recommendations.length - 1 : prev - 1
    );
  };

  const getTypeIcon = () => {
    switch (type) {
      case 'network':
        return <UserGroupIcon className="w-5 h-5" />;
      case 'trending':
        return <SparklesIcon className="w-5 h-5" />;
      case 'similar':
        return <HeartSolid className="w-5 h-5" />;
      default:
        return <SparklesIcon className="w-5 h-5" />;
    }
  };

  const getTypeTitle = () => {
    switch (type) {
      case 'network':
        return 'Stories Your Friends Loved';
      case 'trending':
        return 'Trending in Your Network';
      case 'similar':
        return 'Recommended for You';
      default:
        return 'Recommended Stories';
    }
  };

  const renderRecommendationReasons = (reasons) => {
    if (!reasons || reasons.length === 0) return null;

    return (
      <div className="mt-3 space-y-1">
        {reasons.slice(0, 2).map((reason, index) => (
          <div key={index} className="flex items-center text-xs text-gray-600">
            {reason.type === 'friend_shared' && (
              <>
                <UserGroupIcon className="w-3 h-3 mr-1" />
                <span>Shared by {reason.actorName}</span>
              </>
            )}
            {reason.type === 'topic_match' && (
              <>
                <SparklesIcon className="w-3 h-3 mr-1" />
                <span>Matches your interest in {reason.topicName}</span>
              </>
            )}
            {reason.type === 'high_engagement' && (
              <>
                <HeartIcon className="w-3 h-3 mr-1" />
                <span>High engagement ({Math.round(reason.engagementScore * 100)}%)</span>
              </>
            )}
            {reason.type === 'similar_users' && (
              <>
                <UserGroupIcon className="w-3 h-3 mr-1" />
                <span>Liked by {reason.similarUserCount} similar users</span>
              </>
            )}
          </div>
        ))}
      </div>
    );
  };

  const renderSocialProof = (socialProof) => {
    if (!socialProof) return null;

    return (
      <div className="mt-3 flex items-center space-x-4 text-xs text-gray-500">
        {socialProof.totalShares > 0 && (
          <div className="flex items-center">
            <ShareIcon className="w-3 h-3 mr-1" />
            <span>{socialProof.totalShares} shares</span>
          </div>
        )}
        {socialProof.networkReactions > 0 && (
          <div className="flex items-center">
            <HeartIcon className="w-3 h-3 mr-1" />
            <span>{socialProof.networkReactions} reactions</span>
          </div>
        )}
        {socialProof.networkComments > 0 && (
          <div className="flex items-center">
            <ChatBubbleLeftIcon className="w-3 h-3 mr-1" />
            <span>{socialProof.networkComments} comments</span>
          </div>
        )}
      </div>
    );
  };

  if (loading) {
    return (
      <div className={`bg-white rounded-lg shadow-sm border p-6 ${className}`}>
        <div className="animate-pulse">
          <div className="flex items-center mb-4">
            <div className="w-5 h-5 bg-gray-200 rounded mr-2"></div>
            <div className="h-4 bg-gray-200 rounded w-48"></div>
          </div>
          <div className="space-y-3">
            <div className="h-32 bg-gray-200 rounded"></div>
            <div className="h-4 bg-gray-200 rounded w-3/4"></div>
            <div className="h-3 bg-gray-200 rounded w-1/2"></div>
          </div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className={`bg-white rounded-lg shadow-sm border p-6 ${className}`}>
        <div className="text-center text-red-600">
          <p>Failed to load recommendations</p>
          <button 
            onClick={fetchRecommendations}
            className="mt-2 text-sm text-blue-600 hover:text-blue-800"
          >
            Try again
          </button>
        </div>
      </div>
    );
  }

  if (!recommendations || recommendations.length === 0) {
    return (
      <div className={`bg-white rounded-lg shadow-sm border p-6 ${className}`}>
        <div className="text-center text-gray-500">
          <SparklesIcon className="w-8 h-8 mx-auto mb-2 text-gray-400" />
          <p>No recommendations available</p>
          <p className="text-sm mt-1">Check back later for personalized content</p>
        </div>
      </div>
    );
  }

  const currentRec = recommendations[currentIndex];

  return (
    <div className={`bg-white rounded-lg shadow-sm border overflow-hidden ${className}`}>
      {/* Header */}
      <div className="p-4 border-b bg-gradient-to-r from-blue-50 to-purple-50">
        <div className="flex items-center justify-between">
          <div className="flex items-center">
            <div className="text-blue-600 mr-2">
              {getTypeIcon()}
            </div>
            <h3 className="font-semibold text-gray-900">
              {getTypeTitle()}
            </h3>
          </div>
          <div className="flex items-center space-x-2">
            <span className="text-xs text-gray-500">
              {currentIndex + 1} of {recommendations.length}
            </span>
            {recommendations.length > 1 && (
              <div className="flex space-x-1">
                <button
                  onClick={prevRecommendation}
                  className="p-1 rounded-full hover:bg-white/50 transition-colors"
                >
                  <ChevronLeftIcon className="w-4 h-4 text-gray-600" />
                </button>
                <button
                  onClick={nextRecommendation}
                  className="p-1 rounded-full hover:bg-white/50 transition-colors"
                >
                  <ChevronRightIcon className="w-4 h-4 text-gray-600" />
                </button>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Recommendation Content */}
      <AnimatePresence mode="wait">
        <motion.div
          key={currentIndex}
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          exit={{ opacity: 0, x: -20 }}
          transition={{ duration: 0.3 }}
          className="p-4"
        >
          {/* Article Image */}
          {currentRec.image && (
            <div className="mb-3 rounded-lg overflow-hidden">
              <img
                src={currentRec.image}
                alt={currentRec.title}
                className="w-full h-32 object-cover hover:scale-105 transition-transform duration-300"
                onError={(e) => {
                  e.target.style.display = 'none';
                }}
              />
            </div>
          )}

          {/* Article Content */}
          <div className="space-y-2">
            <a
              href={currentRec.url}
              target="_blank"
              rel="noopener noreferrer"
              className="block group"
            >
              <h4 className="font-semibold text-gray-900 group-hover:text-blue-600 transition-colors line-clamp-2">
                {currentRec.title}
              </h4>
            </a>
            
            <p className="text-sm text-gray-600 line-clamp-2">
              {currentRec.description}
            </p>

            <div className="flex items-center justify-between text-xs text-gray-500">
              <span>{currentRec.author}</span>
              <span>{new Date(currentRec.publishedAt).toLocaleDateString()}</span>
            </div>

            {/* Tags */}
            {currentRec.tags && currentRec.tags.length > 0 && (
              <div className="flex flex-wrap gap-1 mt-2">
                {currentRec.tags.slice(0, 3).map((tag, index) => (
                  <span
                    key={index}
                    className="px-2 py-1 bg-gray-100 text-xs text-gray-600 rounded-full"
                  >
                    {tag}
                  </span>
                ))}
              </div>
            )}

            {/* Recommendation Reasons */}
            {renderRecommendationReasons(currentRec.recommendationReasons)}

            {/* Social Proof */}
            {renderSocialProof(currentRec.socialProof)}

            {/* Recommendation Score */}
            {currentRec.recommendationScore && (
              <div className="mt-3 flex items-center">
                <div className="flex-1 bg-gray-200 rounded-full h-1.5">
                  <div
                    className="bg-gradient-to-r from-blue-500 to-purple-500 h-1.5 rounded-full transition-all duration-500"
                    style={{ width: `${currentRec.recommendationScore * 100}%` }}
                  ></div>
                </div>
                <span className="ml-2 text-xs text-gray-500">
                  {Math.round(currentRec.recommendationScore * 100)}% match
                </span>
              </div>
            )}
          </div>
        </motion.div>
      </AnimatePresence>

      {/* Footer with metadata */}
      {metadata && (
        <div className="px-4 py-2 bg-gray-50 border-t">
          <div className="flex items-center justify-between text-xs text-gray-500">
            <span>Network: {metadata.networkSize} connections</span>
            <span>Topics: {metadata.followedTopics}</span>
          </div>
        </div>
      )}
    </div>
  );
};

export default SocialRecommendations;