import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  UserGroupIcon,
  PlusIcon,
  Cog6ToothIcon,
  MagnifyingGlassIcon,
  UsersIcon,
  ChatBubbleLeftRightIcon,
  PencilIcon,
  TrashIcon,
  EllipsisVerticalIcon,
  PinIcon,
  ExclamationTriangleIcon,
  CheckIcon,
  XMarkIcon
} from '@heroicons/react/24/outline';
import {
  UserGroupIcon as UserGroupSolid,
  PinIcon as PinSolid
} from '@heroicons/react/24/solid';

const GroupManager = ({ userId = 'demo-user-1', className = '' }) => {
  const [groups, setGroups] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState('my-groups'); // 'my-groups', 'discover', 'create'
  const [selectedGroup, setSelectedGroup] = useState(null);
  const [showCreateForm, setShowCreateForm] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');

  // Form states
  const [createForm, setCreateForm] = useState({
    name: '',
    description: '',
    category: 'Technology',
    privacy: 'public',
    tags: []
  });
  const [newTag, setNewTag] = useState('');

  useEffect(() => {
    fetchGroups();
  }, [userId, activeTab]);

  const fetchGroups = async () => {
    try {
      setLoading(true);
      setError(null);
      
      const endpoint = activeTab === 'discover' 
        ? '/api/social/groups?action=discover'
        : '/api/social/groups';
      
      const response = await fetch(endpoint, {
        headers: {
          'user-id': userId
        }
      });
      
      if (!response.ok) {
        throw new Error('Failed to fetch groups');
      }
      
      const data = await response.json();
      setGroups(data.groups || []);
    } catch (err) {
      console.error('Error fetching groups:', err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const createGroup = async () => {
    try {
      const response = await fetch('/api/social/groups', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'user-id': userId
        },
        body: JSON.stringify({
          action: 'create',
          ...createForm
        })
      });
      
      if (!response.ok) {
        throw new Error('Failed to create group');
      }
      
      const data = await response.json();
      setGroups(prev => [data.group, ...prev]);
      setShowCreateForm(false);
      setCreateForm({
        name: '',
        description: '',
        category: 'Technology',
        privacy: 'public',
        tags: []
      });
    } catch (err) {
      console.error('Error creating group:', err);
      setError(err.message);
    }
  };

  const joinGroup = async (groupId) => {
    try {
      const response = await fetch('/api/social/groups', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'user-id': userId
        },
        body: JSON.stringify({
          action: 'join',
          groupId
        })
      });
      
      if (!response.ok) {
        throw new Error('Failed to join group');
      }
      
      // Update local state
      setGroups(prev => prev.map(group => 
        group.id === groupId 
          ? { ...group, isMember: true, memberCount: group.memberCount + 1 }
          : group
      ));
    } catch (err) {
      console.error('Error joining group:', err);
      setError(err.message);
    }
  };

  const leaveGroup = async (groupId) => {
    try {
      const response = await fetch('/api/social/groups', {
        method: 'DELETE',
        headers: {
          'Content-Type': 'application/json',
          'user-id': userId
        },
        body: JSON.stringify({
          action: 'leave',
          groupId
        })
      });
      
      if (!response.ok) {
        throw new Error('Failed to leave group');
      }
      
      // Update local state
      setGroups(prev => prev.map(group => 
        group.id === groupId 
          ? { ...group, isMember: false, memberCount: group.memberCount - 1 }
          : group
      ));
    } catch (err) {
      console.error('Error leaving group:', err);
      setError(err.message);
    }
  };

  const addTag = () => {
    if (newTag.trim() && !createForm.tags.includes(newTag.trim())) {
      setCreateForm(prev => ({
        ...prev,
        tags: [...prev.tags, newTag.trim()]
      }));
      setNewTag('');
    }
  };

  const removeTag = (tagToRemove) => {
    setCreateForm(prev => ({
      ...prev,
      tags: prev.tags.filter(tag => tag !== tagToRemove)
    }));
  };

  const filteredGroups = groups.filter(group => 
    group.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
    group.description.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const renderGroupCard = (group) => (
    <motion.div
      key={group.id}
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="bg-white rounded-lg border shadow-sm hover:shadow-md transition-shadow"
    >
      {/* Group Header */}
      <div className="p-4 border-b">
        <div className="flex items-start justify-between">
          <div className="flex items-center space-x-3">
            <div className="w-12 h-12 bg-gradient-to-br from-blue-500 to-purple-600 rounded-lg flex items-center justify-center">
              <UserGroupSolid className="w-6 h-6 text-white" />
            </div>
            <div>
              <h3 className="font-semibold text-gray-900">{group.name}</h3>
              <p className="text-sm text-gray-600">{group.category}</p>
              <div className="flex items-center space-x-3 mt-1 text-xs text-gray-500">
                <span className="flex items-center">
                  <UsersIcon className="w-3 h-3 mr-1" />
                  {group.memberCount} members
                </span>
                <span className="flex items-center">
                  <ChatBubbleLeftRightIcon className="w-3 h-3 mr-1" />
                  {group.postCount} posts
                </span>
                {group.privacy === 'private' && (
                  <span className="px-2 py-0.5 bg-yellow-100 text-yellow-800 rounded-full text-xs">
                    Private
                  </span>
                )}
              </div>
            </div>
          </div>
          
          {/* Action Button */}
          <div className="flex items-center space-x-2">
            {group.isMember ? (
              <>
                <button
                  onClick={() => setSelectedGroup(group)}
                  className="px-3 py-1.5 text-sm bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors"
                >
                  View
                </button>
                <button
                  onClick={() => leaveGroup(group.id)}
                  className="px-3 py-1.5 text-sm border border-gray-300 text-gray-700 rounded-md hover:bg-gray-50 transition-colors"
                >
                  Leave
                </button>
              </>
            ) : (
              <button
                onClick={() => joinGroup(group.id)}
                className="px-3 py-1.5 text-sm bg-green-600 text-white rounded-md hover:bg-green-700 transition-colors"
              >
                Join
              </button>
            )}
          </div>
        </div>
        
        <p className="mt-3 text-sm text-gray-600 line-clamp-2">
          {group.description}
        </p>
        
        {/* Tags */}
        {group.tags && group.tags.length > 0 && (
          <div className="flex flex-wrap gap-1 mt-3">
            {group.tags.slice(0, 3).map((tag, index) => (
              <span
                key={index}
                className="px-2 py-1 bg-gray-100 text-xs text-gray-600 rounded-full"
              >
                {tag}
              </span>
            ))}
            {group.tags.length > 3 && (
              <span className="px-2 py-1 bg-gray-100 text-xs text-gray-600 rounded-full">
                +{group.tags.length - 3} more
              </span>
            )}
          </div>
        )}
      </div>
      
      {/* Recent Activity */}
      {group.recentPosts && group.recentPosts.length > 0 && (
        <div className="p-4">
          <h4 className="text-sm font-medium text-gray-900 mb-2">Recent Activity</h4>
          <div className="space-y-2">
            {group.recentPosts.slice(0, 2).map((post, index) => (
              <div key={index} className="text-sm">
                <p className="text-gray-900 line-clamp-1">{post.title}</p>
                <p className="text-gray-500 text-xs">
                  by {post.author} • {new Date(post.createdAt).toLocaleDateString()}
                </p>
              </div>
            ))}
          </div>
        </div>
      )}
    </motion.div>
  );

  const renderCreateForm = () => (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      className="bg-white rounded-lg border shadow-lg p-6"
    >
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-gray-900">Create New Group</h3>
        <button
          onClick={() => setShowCreateForm(false)}
          className="text-gray-400 hover:text-gray-600"
        >
          <XMarkIcon className="w-5 h-5" />
        </button>
      </div>
      
      <div className="space-y-4">
        {/* Group Name */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Group Name
          </label>
          <input
            type="text"
            value={createForm.name}
            onChange={(e) => setCreateForm(prev => ({ ...prev, name: e.target.value }))}
            className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            placeholder="Enter group name"
          />
        </div>
        
        {/* Description */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Description
          </label>
          <textarea
            value={createForm.description}
            onChange={(e) => setCreateForm(prev => ({ ...prev, description: e.target.value }))}
            rows={3}
            className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            placeholder="Describe your group"
          />
        </div>
        
        {/* Category and Privacy */}
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Category
            </label>
            <select
              value={createForm.category}
              onChange={(e) => setCreateForm(prev => ({ ...prev, category: e.target.value }))}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value="Technology">Technology</option>
              <option value="Science">Science</option>
              <option value="Business">Business</option>
              <option value="Health">Health</option>
              <option value="Environment">Environment</option>
              <option value="Politics">Politics</option>
              <option value="Sports">Sports</option>
              <option value="Entertainment">Entertainment</option>
            </select>
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Privacy
            </label>
            <select
              value={createForm.privacy}
              onChange={(e) => setCreateForm(prev => ({ ...prev, privacy: e.target.value }))}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value="public">Public</option>
              <option value="private">Private</option>
            </select>
          </div>
        </div>
        
        {/* Tags */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Tags
          </label>
          <div className="flex space-x-2 mb-2">
            <input
              type="text"
              value={newTag}
              onChange={(e) => setNewTag(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && addTag()}
              className="flex-1 px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              placeholder="Add a tag"
            />
            <button
              onClick={addTag}
              className="px-3 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors"
            >
              Add
            </button>
          </div>
          
          {createForm.tags.length > 0 && (
            <div className="flex flex-wrap gap-1">
              {createForm.tags.map((tag, index) => (
                <span
                  key={index}
                  className="inline-flex items-center px-2 py-1 bg-blue-100 text-blue-800 text-xs rounded-full"
                >
                  {tag}
                  <button
                    onClick={() => removeTag(tag)}
                    className="ml-1 text-blue-600 hover:text-blue-800"
                  >
                    <XMarkIcon className="w-3 h-3" />
                  </button>
                </span>
              ))}
            </div>
          )}
        </div>
        
        {/* Actions */}
        <div className="flex justify-end space-x-3 pt-4">
          <button
            onClick={() => setShowCreateForm(false)}
            className="px-4 py-2 border border-gray-300 text-gray-700 rounded-md hover:bg-gray-50 transition-colors"
          >
            Cancel
          </button>
          <button
            onClick={createGroup}
            disabled={!createForm.name.trim() || !createForm.description.trim()}
            className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            Create Group
          </button>
        </div>
      </div>
    </motion.div>
  );

  if (loading) {
    return (
      <div className={`bg-white rounded-lg shadow-sm border p-6 ${className}`}>
        <div className="animate-pulse space-y-4">
          <div className="h-6 bg-gray-200 rounded w-1/3"></div>
          <div className="space-y-3">
            {[1, 2, 3].map(i => (
              <div key={i} className="border rounded-lg p-4">
                <div className="flex items-center space-x-3 mb-3">
                  <div className="w-12 h-12 bg-gray-200 rounded-lg"></div>
                  <div className="flex-1">
                    <div className="h-4 bg-gray-200 rounded w-1/2 mb-2"></div>
                    <div className="h-3 bg-gray-200 rounded w-1/3"></div>
                  </div>
                </div>
                <div className="h-3 bg-gray-200 rounded w-full mb-2"></div>
                <div className="h-3 bg-gray-200 rounded w-2/3"></div>
              </div>
            ))}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className={`bg-white rounded-lg shadow-sm border ${className}`}>
      {/* Header */}
      <div className="p-6 border-b">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-semibold text-gray-900 flex items-center">
            <UserGroupIcon className="w-6 h-6 mr-2 text-blue-600" />
            Groups & Communities
          </h2>
          <button
            onClick={() => setShowCreateForm(true)}
            className="flex items-center px-3 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors"
          >
            <PlusIcon className="w-4 h-4 mr-1" />
            Create Group
          </button>
        </div>
        
        {/* Tabs */}
        <div className="flex space-x-1 bg-gray-100 rounded-lg p-1">
          {[
            { id: 'my-groups', label: 'My Groups' },
            { id: 'discover', label: 'Discover' }
          ].map(tab => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`flex-1 px-3 py-2 text-sm font-medium rounded-md transition-colors ${
                activeTab === tab.id
                  ? 'bg-white text-blue-600 shadow-sm'
                  : 'text-gray-600 hover:text-gray-900'
              }`}
            >
              {tab.label}
            </button>
          ))}
        </div>
        
        {/* Search */}
        <div className="mt-4 relative">
          <MagnifyingGlassIcon className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search groups..."
            className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
        </div>
      </div>
      
      {/* Content */}
      <div className="p-6">
        {showCreateForm && (
          <div className="mb-6">
            {renderCreateForm()}
          </div>
        )}
        
        {error && (
          <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded-md">
            <div className="flex items-center">
              <ExclamationTriangleIcon className="w-5 h-5 text-red-400 mr-2" />
              <span className="text-red-800">{error}</span>
            </div>
          </div>
        )}
        
        {filteredGroups.length === 0 ? (
          <div className="text-center py-8">
            <UserGroupIcon className="w-12 h-12 text-gray-400 mx-auto mb-3" />
            <p className="text-gray-600">
              {activeTab === 'my-groups' 
                ? 'You haven\'t joined any groups yet'
                : 'No groups found'
              }
            </p>
            <p className="text-sm text-gray-500 mt-1">
              {activeTab === 'my-groups' 
                ? 'Discover and join groups to connect with like-minded people'
                : 'Try adjusting your search terms'
              }
            </p>
          </div>
        ) : (
          <div className="grid gap-4">
            {filteredGroups.map(renderGroupCard)}
          </div>
        )}
      </div>
    </div>
  );
};

export default GroupManager;