import React, { useState, useEffect } from 'react';
import { Button } from './ui/button';
import { Avatar, AvatarFallback, AvatarImage } from './ui/avatar';
import { Badge } from './ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import { motion } from 'framer-motion';
import { 
  User, 
  MapPin, 
  Calendar, 
  Link as LinkIcon, 
  Edit3,
  Settings,
  Share2,
  Heart,
  MessageSquare,
  Users,
  TrendingUp,
  BookOpen,
  Award
} from 'lucide-react';
import FollowButton from './FollowButton';
import NewsCard from './NewsCard';

/**
 * Activity Card Component
 */
const ActivityCard = ({ activity }) => {
  const getActivityIcon = () => {
    switch (activity.type) {
      case 'share':
        return <Share2 className="h-4 w-4 text-blue-500" />;
      case 'reaction':
        return <Heart className="h-4 w-4 text-red-500" />;
      case 'comment':
        return <MessageSquare className="h-4 w-4 text-green-500" />;
      case 'follow':
        return <Users className="h-4 w-4 text-purple-500" />;
      default:
        return <TrendingUp className="h-4 w-4 text-gray-500" />;
    }
  };

  const getActivityText = () => {
    switch (activity.type) {
      case 'share':
        return `shared an article: "${activity.article?.title}"`;
      case 'reaction':
        return `reacted ${activity.reaction} to "${activity.article?.title}"`;
      case 'comment':
        return `commented on "${activity.article?.title}"`;
      case 'follow':
        return `started following ${activity.target?.name}`;
      default:
        return 'had some activity';
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="flex gap-3 p-4 border border-gray-200 rounded-lg hover:bg-gray-50 transition-colors"
    >
      <div className="flex-shrink-0 mt-1">
        {getActivityIcon()}
      </div>
      
      <div className="flex-1 space-y-2">
        <p className="text-sm text-gray-700">
          <span className="font-medium">{activity.user?.name}</span>
          {' '}
          {getActivityText()}
        </p>
        
        {activity.note && (
          <p className="text-sm text-gray-600 italic bg-gray-100 p-2 rounded">
            "{activity.note}"
          </p>
        )}
        
        {activity.article && (
          <div className="mt-2">
            <a 
              href={activity.article.url}
              target="_blank"
              rel="noopener noreferrer"
              className="text-blue-600 hover:text-blue-800 text-sm font-medium"
            >
              {activity.article.title}
            </a>
          </div>
        )}
        
        <div className="flex items-center gap-4 text-xs text-gray-500">
          <span>{new Date(activity.createdAt).toLocaleDateString()}</span>
          {activity.engagement && (
            <span>{activity.engagement.likes} likes</span>
          )}
        </div>
      </div>
    </motion.div>
  );
};

/**
 * Stats Card Component
 */
const StatsCard = ({ icon: Icon, label, value, color = 'text-gray-600' }) => (
  <div className="text-center p-4 bg-white rounded-lg border border-gray-200">
    <Icon className={`h-6 w-6 mx-auto mb-2 ${color}`} />
    <div className="text-2xl font-bold text-gray-900">{value}</div>
    <div className="text-sm text-gray-600">{label}</div>
  </div>
);

/**
 * UserProfile Component
 * Displays user profile with avatar, bio, interests, stats, and activity stream
 */
const UserProfile = ({ 
  userId, 
  isOwnProfile = false, 
  currentUserId 
}) => {
  const [user, setUser] = useState(null);
  const [activities, setActivities] = useState([]);
  const [sharedArticles, setSharedArticles] = useState([]);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState('activity');
  const [isFollowing, setIsFollowing] = useState(false);

  useEffect(() => {
    fetchUserProfile();
    fetchUserActivities();
    fetchSharedArticles();
  }, [userId]);

  const fetchUserProfile = async () => {
    try {
      const response = await fetch(`/api/social/users/${userId}`);
      if (response.ok) {
        const data = await response.json();
        setUser(data.user);
        setIsFollowing(data.isFollowing || false);
      }
    } catch (error) {
      console.error('Error fetching user profile:', error);
    }
  };

  const fetchUserActivities = async () => {
    try {
      const response = await fetch(`/api/social/users/${userId}/activities`);
      if (response.ok) {
        const data = await response.json();
        setActivities(data.activities || []);
      }
    } catch (error) {
      console.error('Error fetching user activities:', error);
    }
  };

  const fetchSharedArticles = async () => {
    try {
      const response = await fetch(`/api/social/users/${userId}/shares`);
      if (response.ok) {
        const data = await response.json();
        setSharedArticles(data.articles || []);
      }
    } catch (error) {
      console.error('Error fetching shared articles:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleFollowChange = (newState) => {
    setIsFollowing(newState);
    setUser(prev => ({
      ...prev,
      stats: {
        ...prev.stats,
        followers: prev.stats.followers + (newState ? 1 : -1)
      }
    }));
  };

  if (loading) {
    return (
      <div className="flex justify-center items-center h-64">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
      </div>
    );
  }

  if (!user) {
    return (
      <div className="text-center py-12">
        <User className="h-12 w-12 mx-auto mb-4 text-gray-400" />
        <h3 className="text-lg font-medium text-gray-900 mb-2">User not found</h3>
        <p className="text-gray-600">The user profile you're looking for doesn't exist.</p>
      </div>
    );
  }

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      {/* Profile Header */}
      <div className="bg-white rounded-lg border border-gray-200 p-6">
        <div className="flex flex-col md:flex-row gap-6">
          {/* Avatar and Basic Info */}
          <div className="flex flex-col items-center md:items-start">
            <Avatar className="h-24 w-24 mb-4">
              <AvatarImage src={user.avatar} />
              <AvatarFallback className="text-2xl">
                {user.name?.charAt(0) || 'U'}
              </AvatarFallback>
            </Avatar>
            
            <div className="text-center md:text-left">
              <h1 className="text-2xl font-bold text-gray-900 mb-1">{user.name}</h1>
              <p className="text-gray-600 mb-2">@{user.username}</p>
              
              {user.location && (
                <div className="flex items-center gap-1 text-sm text-gray-500 mb-1">
                  <MapPin className="h-4 w-4" />
                  <span>{user.location}</span>
                </div>
              )}
              
              <div className="flex items-center gap-1 text-sm text-gray-500">
                <Calendar className="h-4 w-4" />
                <span>Joined {new Date(user.createdAt).toLocaleDateString()}</span>
              </div>
            </div>
          </div>
          
          {/* Bio and Actions */}
          <div className="flex-1">
            {user.bio && (
              <p className="text-gray-700 mb-4 leading-relaxed">{user.bio}</p>
            )}
            
            {/* Interests */}
            {user.interests && user.interests.length > 0 && (
              <div className="mb-4">
                <h3 className="text-sm font-medium text-gray-900 mb-2">Interests</h3>
                <div className="flex flex-wrap gap-2">
                  {user.interests.map((interest, index) => (
                    <Badge key={index} variant="secondary">
                      {interest}
                    </Badge>
                  ))}
                </div>
              </div>
            )}
            
            {/* Website */}
            {user.website && (
              <div className="mb-4">
                <a 
                  href={user.website}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="flex items-center gap-1 text-blue-600 hover:text-blue-800 text-sm"
                >
                  <LinkIcon className="h-4 w-4" />
                  <span>{user.website}</span>
                </a>
              </div>
            )}
            
            {/* Action Buttons */}
            <div className="flex gap-3">
              {!isOwnProfile && (
                <FollowButton
                  type="user"
                  targetId={userId}
                  targetName={user.name}
                  isFollowing={isFollowing}
                  onFollowChange={handleFollowChange}
                  size="md"
                />
              )}
              
              {isOwnProfile && (
                <Button variant="outline" size="md">
                  <Edit3 className="h-4 w-4 mr-2" />
                  Edit Profile
                </Button>
              )}
              
              <Button variant="ghost" size="md">
                <Settings className="h-4 w-4" />
              </Button>
            </div>
          </div>
        </div>
      </div>
      
      {/* Stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <StatsCard 
          icon={Users} 
          label="Followers" 
          value={user.stats?.followers || 0}
          color="text-blue-600"
        />
        <StatsCard 
          icon={Users} 
          label="Following" 
          value={user.stats?.following || 0}
          color="text-green-600"
        />
        <StatsCard 
          icon={Share2} 
          label="Articles Shared" 
          value={user.stats?.articlesShared || 0}
          color="text-purple-600"
        />
        <StatsCard 
          icon={Award} 
          label="Reputation" 
          value={user.stats?.reputation || 0}
          color="text-orange-600"
        />
      </div>
      
      {/* Content Tabs */}
      <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="activity">Recent Activity</TabsTrigger>
          <TabsTrigger value="shares">Shared Articles</TabsTrigger>
          <TabsTrigger value="interests">Topics</TabsTrigger>
        </TabsList>
        
        <TabsContent value="activity" className="space-y-4">
          <div className="bg-white rounded-lg border border-gray-200 p-6">
            <h2 className="text-lg font-semibold text-gray-900 mb-4">Recent Activity</h2>
            
            {activities.length > 0 ? (
              <div className="space-y-4">
                {activities.map((activity) => (
                  <ActivityCard key={activity.id} activity={activity} />
                ))}
              </div>
            ) : (
              <div className="text-center py-8 text-gray-500">
                <TrendingUp className="h-12 w-12 mx-auto mb-3 text-gray-300" />
                <p>No recent activity to show.</p>
              </div>
            )}
          </div>
        </TabsContent>
        
        <TabsContent value="shares" className="space-y-4">
          <div className="bg-white rounded-lg border border-gray-200 p-6">
            <h2 className="text-lg font-semibold text-gray-900 mb-4">Shared Articles</h2>
            
            {sharedArticles.length > 0 ? (
              <div className="grid gap-4">
                {sharedArticles.map((article) => (
                  <NewsCard 
                    key={article.id} 
                    article={article}
                    showSocialFeatures={true}
                  />
                ))}
              </div>
            ) : (
              <div className="text-center py-8 text-gray-500">
                <BookOpen className="h-12 w-12 mx-auto mb-3 text-gray-300" />
                <p>No articles shared yet.</p>
              </div>
            )}
          </div>
        </TabsContent>
        
        <TabsContent value="interests" className="space-y-4">
          <div className="bg-white rounded-lg border border-gray-200 p-6">
            <h2 className="text-lg font-semibold text-gray-900 mb-4">Followed Topics</h2>
            
            {user.followedTopics && user.followedTopics.length > 0 ? (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {user.followedTopics.map((topic) => (
                  <div key={topic.id} className="p-4 border border-gray-200 rounded-lg">
                    <h3 className="font-medium text-gray-900 mb-2">{topic.name}</h3>
                    <p className="text-sm text-gray-600 mb-3">{topic.description}</p>
                    <div className="flex justify-between items-center">
                      <span className="text-xs text-gray-500">
                        {topic.articleCount} articles
                      </span>
                      <FollowButton
                        type="topic"
                        targetId={topic.id}
                        targetName={topic.name}
                        isFollowing={true}
                        size="sm"
                      />
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-center py-8 text-gray-500">
                <BookOpen className="h-12 w-12 mx-auto mb-3 text-gray-300" />
                <p>No topics followed yet.</p>
              </div>
            )}
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default UserProfile;