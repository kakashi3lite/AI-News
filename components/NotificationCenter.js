import React, { useState, useEffect } from 'react';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { Avatar, AvatarFallback, AvatarImage } from './ui/avatar';
import { Popover, PopoverContent, PopoverTrigger } from './ui/popover';
import { Switch } from './ui/switch';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Bell, 
  BellOff,
  Settings,
  Check,
  X,
  Heart,
  MessageSquare,
  UserPlus,
  Share2,
  Users,
  TrendingUp,
  Trash2,
  MoreHorizontal
} from 'lucide-react';

/**
 * Individual Notification Component
 */
const NotificationItem = ({ notification, onMarkAsRead, onDelete }) => {
  const getNotificationIcon = () => {
    switch (notification.type) {
      case 'reaction':
        return <Heart className="h-4 w-4 text-red-500" />;
      case 'comment':
        return <MessageSquare className="h-4 w-4 text-blue-500" />;
      case 'follow':
        return <UserPlus className="h-4 w-4 text-green-500" />;
      case 'share':
        return <Share2 className="h-4 w-4 text-purple-500" />;
      case 'group_post':
        return <Users className="h-4 w-4 text-orange-500" />;
      case 'trending':
        return <TrendingUp className="h-4 w-4 text-yellow-500" />;
      default:
        return <Bell className="h-4 w-4 text-gray-500" />;
    }
  };

  const getNotificationText = () => {
    switch (notification.type) {
      case 'reaction':
        return `${notification.actor?.name} reacted ${notification.data?.reaction} to your ${notification.data?.targetType}`;
      case 'comment':
        return `${notification.actor?.name} commented on your article`;
      case 'follow':
        return `${notification.actor?.name} started following you`;
      case 'share':
        return `${notification.actor?.name} shared your article`;
      case 'group_post':
        return `New post in ${notification.data?.groupName}`;
      case 'trending':
        return `Your article "${notification.data?.articleTitle}" is trending`;
      default:
        return notification.message || 'You have a new notification';
    }
  };

  const handleClick = () => {
    if (!notification.isRead) {
      onMarkAsRead(notification.id);
    }
    
    // Navigate to relevant content
    if (notification.data?.url) {
      window.open(notification.data.url, '_blank');
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0, x: -20 }}
      animate={{ opacity: 1, x: 0 }}
      exit={{ opacity: 0, x: 20 }}
      className={`
        flex gap-3 p-4 border-b border-gray-100 hover:bg-gray-50 cursor-pointer transition-colors
        ${!notification.isRead ? 'bg-blue-50 border-l-4 border-l-blue-500' : ''}
      `}
      onClick={handleClick}
    >
      {/* Actor Avatar */}
      <Avatar className="h-8 w-8 flex-shrink-0">
        <AvatarImage src={notification.actor?.avatar} />
        <AvatarFallback>
          {notification.actor?.name?.charAt(0) || getNotificationIcon()}
        </AvatarFallback>
      </Avatar>
      
      {/* Content */}
      <div className="flex-1 space-y-1">
        <div className="flex items-start justify-between">
          <p className="text-sm text-gray-900 leading-relaxed">
            {getNotificationText()}
          </p>
          
          <div className="flex items-center gap-2 ml-2">
            {!notification.isRead && (
              <div className="h-2 w-2 bg-blue-500 rounded-full"></div>
            )}
            
            <Popover>
              <PopoverTrigger asChild>
                <Button
                  variant="ghost"
                  size="sm"
                  className="h-6 w-6 p-0 text-gray-400 hover:text-gray-600"
                  onClick={(e) => e.stopPropagation()}
                >
                  <MoreHorizontal className="h-3 w-3" />
                </Button>
              </PopoverTrigger>
              <PopoverContent className="w-40 p-1" align="end">
                <div className="space-y-1">
                  {!notification.isRead && (
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={(e) => {
                        e.stopPropagation();
                        onMarkAsRead(notification.id);
                      }}
                      className="w-full justify-start h-8 px-2"
                    >
                      <Check className="h-3 w-3 mr-2" />
                      Mark as read
                    </Button>
                  )}
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={(e) => {
                      e.stopPropagation();
                      onDelete(notification.id);
                    }}
                    className="w-full justify-start h-8 px-2 text-red-600 hover:text-red-700"
                  >
                    <Trash2 className="h-3 w-3 mr-2" />
                    Delete
                  </Button>
                </div>
              </PopoverContent>
            </Popover>
          </div>
        </div>
        
        {/* Additional content */}
        {notification.data?.preview && (
          <p className="text-xs text-gray-600 bg-gray-100 p-2 rounded">
            "{notification.data.preview}"
          </p>
        )}
        
        <div className="flex items-center justify-between">
          <span className="text-xs text-gray-500">
            {new Date(notification.createdAt).toLocaleDateString()}
          </span>
          
          {notification.type && (
            <div className="flex items-center gap-1">
              {getNotificationIcon()}
            </div>
          )}
        </div>
      </div>
    </motion.div>
  );
};

/**
 * Notification Settings Component
 */
const NotificationSettings = ({ settings, onSettingsChange }) => {
  const notificationTypes = [
    {
      id: 'reactions',
      label: 'Reactions',
      description: 'When someone reacts to your content',
      icon: Heart
    },
    {
      id: 'comments',
      label: 'Comments',
      description: 'When someone comments on your articles',
      icon: MessageSquare
    },
    {
      id: 'follows',
      label: 'New Followers',
      description: 'When someone follows you',
      icon: UserPlus
    },
    {
      id: 'shares',
      label: 'Shares',
      description: 'When someone shares your content',
      icon: Share2
    },
    {
      id: 'group_posts',
      label: 'Group Posts',
      description: 'New posts in groups you\'re part of',
      icon: Users
    },
    {
      id: 'trending',
      label: 'Trending Content',
      description: 'When your content is trending',
      icon: TrendingUp
    }
  ];

  return (
    <div className="space-y-6">
      <div>
        <h3 className="text-lg font-semibold text-gray-900 mb-2">Notification Preferences</h3>
        <p className="text-sm text-gray-600">Choose what notifications you want to receive.</p>
      </div>
      
      <div className="space-y-4">
        {notificationTypes.map((type) => {
          const Icon = type.icon;
          return (
            <div key={type.id} className="flex items-center justify-between p-4 border border-gray-200 rounded-lg">
              <div className="flex items-center gap-3">
                <Icon className="h-5 w-5 text-gray-600" />
                <div>
                  <h4 className="font-medium text-gray-900">{type.label}</h4>
                  <p className="text-sm text-gray-600">{type.description}</p>
                </div>
              </div>
              
              <Switch
                checked={settings[type.id] !== false}
                onCheckedChange={(checked) => onSettingsChange(type.id, checked)}
              />
            </div>
          );
        })}
      </div>
      
      <div className="pt-4 border-t border-gray-200">
        <div className="flex items-center justify-between">
          <div>
            <h4 className="font-medium text-gray-900">Email Notifications</h4>
            <p className="text-sm text-gray-600">Receive notifications via email</p>
          </div>
          <Switch
            checked={settings.emailNotifications !== false}
            onCheckedChange={(checked) => onSettingsChange('emailNotifications', checked)}
          />
        </div>
      </div>
    </div>
  );
};

/**
 * NotificationCenter Component
 * Main component for displaying and managing notifications
 */
const NotificationCenter = ({ currentUserId }) => {
  const [notifications, setNotifications] = useState([]);
  const [settings, setSettings] = useState({});
  const [loading, setLoading] = useState(true);
  const [isOpen, setIsOpen] = useState(false);
  const [activeTab, setActiveTab] = useState('all');
  const [unreadCount, setUnreadCount] = useState(0);

  useEffect(() => {
    fetchNotifications();
    fetchSettings();
    
    // Set up real-time notifications (WebSocket or polling)
    const interval = setInterval(fetchNotifications, 30000); // Poll every 30 seconds
    
    return () => clearInterval(interval);
  }, [currentUserId]);

  const fetchNotifications = async () => {
    try {
      const response = await fetch('/api/social/notifications');
      if (response.ok) {
        const data = await response.json();
        setNotifications(data.notifications || []);
        setUnreadCount(data.unreadCount || 0);
      }
    } catch (error) {
      console.error('Error fetching notifications:', error);
    } finally {
      setLoading(false);
    }
  };

  const fetchSettings = async () => {
    try {
      const response = await fetch('/api/social/notifications/settings');
      if (response.ok) {
        const data = await response.json();
        setSettings(data.settings || {});
      }
    } catch (error) {
      console.error('Error fetching notification settings:', error);
    }
  };

  const handleMarkAsRead = async (notificationId) => {
    try {
      const response = await fetch(`/api/social/notifications/${notificationId}/read`, {
        method: 'POST'
      });
      
      if (response.ok) {
        setNotifications(prev => 
          prev.map(notif => 
            notif.id === notificationId 
              ? { ...notif, isRead: true }
              : notif
          )
        );
        setUnreadCount(prev => Math.max(0, prev - 1));
      }
    } catch (error) {
      console.error('Error marking notification as read:', error);
    }
  };

  const handleMarkAllAsRead = async () => {
    try {
      const response = await fetch('/api/social/notifications/read-all', {
        method: 'POST'
      });
      
      if (response.ok) {
        setNotifications(prev => 
          prev.map(notif => ({ ...notif, isRead: true }))
        );
        setUnreadCount(0);
      }
    } catch (error) {
      console.error('Error marking all notifications as read:', error);
    }
  };

  const handleDelete = async (notificationId) => {
    try {
      const response = await fetch(`/api/social/notifications/${notificationId}`, {
        method: 'DELETE'
      });
      
      if (response.ok) {
        const deletedNotification = notifications.find(n => n.id === notificationId);
        setNotifications(prev => prev.filter(notif => notif.id !== notificationId));
        
        if (deletedNotification && !deletedNotification.isRead) {
          setUnreadCount(prev => Math.max(0, prev - 1));
        }
      }
    } catch (error) {
      console.error('Error deleting notification:', error);
    }
  };

  const handleSettingsChange = async (settingKey, value) => {
    const newSettings = { ...settings, [settingKey]: value };
    setSettings(newSettings);
    
    try {
      await fetch('/api/social/notifications/settings', {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ settings: newSettings })
      });
    } catch (error) {
      console.error('Error updating notification settings:', error);
    }
  };

  const getFilteredNotifications = () => {
    switch (activeTab) {
      case 'unread':
        return notifications.filter(n => !n.isRead);
      case 'mentions':
        return notifications.filter(n => ['comment', 'reaction'].includes(n.type));
      case 'follows':
        return notifications.filter(n => n.type === 'follow');
      default:
        return notifications;
    }
  };

  return (
    <Popover open={isOpen} onOpenChange={setIsOpen}>
      <PopoverTrigger asChild>
        <Button
          variant="ghost"
          size="sm"
          className="relative p-2"
          title="Notifications"
        >
          <Bell className="h-5 w-5" />
          {unreadCount > 0 && (
            <Badge 
              variant="destructive" 
              className="absolute -top-1 -right-1 h-5 w-5 flex items-center justify-center p-0 text-xs"
            >
              {unreadCount > 99 ? '99+' : unreadCount}
            </Badge>
          )}
        </Button>
      </PopoverTrigger>
      
      <PopoverContent className="w-96 p-0" align="end">
        <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
          <div className="p-4 border-b border-gray-200">
            <div className="flex items-center justify-between mb-3">
              <h2 className="text-lg font-semibold text-gray-900">Notifications</h2>
              
              <div className="flex items-center gap-2">
                {unreadCount > 0 && (
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={handleMarkAllAsRead}
                    className="text-xs"
                  >
                    Mark all read
                  </Button>
                )}
                
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => setActiveTab('settings')}
                  className="p-1"
                >
                  <Settings className="h-4 w-4" />
                </Button>
              </div>
            </div>
            
            <TabsList className="grid w-full grid-cols-4">
              <TabsTrigger value="all" className="text-xs">All</TabsTrigger>
              <TabsTrigger value="unread" className="text-xs">Unread</TabsTrigger>
              <TabsTrigger value="mentions" className="text-xs">Mentions</TabsTrigger>
              <TabsTrigger value="follows" className="text-xs">Follows</TabsTrigger>
            </TabsList>
          </div>
          
          <TabsContent value="all" className="m-0">
            <div className="max-h-96 overflow-y-auto">
              <AnimatePresence>
                {getFilteredNotifications().map((notification) => (
                  <NotificationItem
                    key={notification.id}
                    notification={notification}
                    onMarkAsRead={handleMarkAsRead}
                    onDelete={handleDelete}
                  />
                ))}
              </AnimatePresence>
              
              {getFilteredNotifications().length === 0 && (
                <div className="text-center py-8 text-gray-500">
                  <Bell className="h-12 w-12 mx-auto mb-3 text-gray-300" />
                  <p>No notifications yet.</p>
                </div>
              )}
            </div>
          </TabsContent>
          
          <TabsContent value="unread" className="m-0">
            <div className="max-h-96 overflow-y-auto">
              <AnimatePresence>
                {getFilteredNotifications().map((notification) => (
                  <NotificationItem
                    key={notification.id}
                    notification={notification}
                    onMarkAsRead={handleMarkAsRead}
                    onDelete={handleDelete}
                  />
                ))}
              </AnimatePresence>
              
              {getFilteredNotifications().length === 0 && (
                <div className="text-center py-8 text-gray-500">
                  <Check className="h-12 w-12 mx-auto mb-3 text-gray-300" />
                  <p>All caught up!</p>
                </div>
              )}
            </div>
          </TabsContent>
          
          <TabsContent value="mentions" className="m-0">
            <div className="max-h-96 overflow-y-auto">
              <AnimatePresence>
                {getFilteredNotifications().map((notification) => (
                  <NotificationItem
                    key={notification.id}
                    notification={notification}
                    onMarkAsRead={handleMarkAsRead}
                    onDelete={handleDelete}
                  />
                ))}
              </AnimatePresence>
              
              {getFilteredNotifications().length === 0 && (
                <div className="text-center py-8 text-gray-500">
                  <MessageSquare className="h-12 w-12 mx-auto mb-3 text-gray-300" />
                  <p>No mentions yet.</p>
                </div>
              )}
            </div>
          </TabsContent>
          
          <TabsContent value="follows" className="m-0">
            <div className="max-h-96 overflow-y-auto">
              <AnimatePresence>
                {getFilteredNotifications().map((notification) => (
                  <NotificationItem
                    key={notification.id}
                    notification={notification}
                    onMarkAsRead={handleMarkAsRead}
                    onDelete={handleDelete}
                  />
                ))}
              </AnimatePresence>
              
              {getFilteredNotifications().length === 0 && (
                <div className="text-center py-8 text-gray-500">
                  <UserPlus className="h-12 w-12 mx-auto mb-3 text-gray-300" />
                  <p>No new followers yet.</p>
                </div>
              )}
            </div>
          </TabsContent>
          
          <TabsContent value="settings" className="m-0">
            <div className="p-4">
              <NotificationSettings
                settings={settings}
                onSettingsChange={handleSettingsChange}
              />
            </div>
          </TabsContent>
        </Tabs>
      </PopoverContent>
    </Popover>
  );
};

export default NotificationCenter;