import React, { useState, useEffect, useRef, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Search, Filter, Clock, TrendingUp, X, ArrowRight, Sparkles } from 'lucide-react';
import { Input } from './ui/input';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { Card } from './ui/card';

const SearchOverlay = ({ 
  isOpen, 
  onClose, 
  onSearch, 
  onFilter, 
  recentSearches = [], 
  trendingTopics = [],
  currentQuery = '',
  currentCategory = ''
}) => {
  const [query, setQuery] = useState('');
  const [suggestions, setSuggestions] = useState([]);
  const [loading, setLoading] = useState(false);
  const [activeSection, setActiveSection] = useState('search'); // 'search', 'filters', 'recent'
  const inputRef = useRef(null);
  const overlayRef = useRef(null);

  // Quick filter categories
  const quickFilters = [
    { id: 'general', label: 'General', icon: '📰' },
    { id: 'technology', label: 'Technology', icon: '💻' },
    { id: 'business', label: 'Business', icon: '💼' },
    { id: 'science', label: 'Science', icon: '🔬' },
    { id: 'health', label: 'Health', icon: '🏥' },
    { id: 'sports', label: 'Sports', icon: '⚽' },
    { id: 'entertainment', label: 'Entertainment', icon: '🎬' },
    { id: 'politics', label: 'Politics', icon: '🏛️' }
  ];

  // Focus input when overlay opens
  useEffect(() => {
    if (isOpen && inputRef.current) {
      setTimeout(() => inputRef.current.focus(), 100);
    }
  }, [isOpen]);

  // Debounced search suggestions
  const fetchSuggestions = useCallback(async (searchQuery) => {
    if (!searchQuery.trim()) {
      setSuggestions([]);
      return;
    }

    setLoading(true);
    try {
      const response = await fetch('/api/search-suggestions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: searchQuery })
      });
      const data = await response.json();
      setSuggestions(data.suggestions || []);
    } catch (error) {
      console.error('Failed to fetch suggestions:', error);
      setSuggestions([]);
    }
    setLoading(false);
  }, []);

  // Debounce suggestions
  useEffect(() => {
    const timer = setTimeout(() => {
      fetchSuggestions(query);
    }, 300);
    return () => clearTimeout(timer);
  }, [query, fetchSuggestions]);

  // Handle search submission
  const handleSearch = (searchQuery = query) => {
    if (searchQuery.trim()) {
      onSearch(searchQuery.trim());
      onClose();
    }
  };

  // Handle filter selection
  const handleFilterSelect = (filterId) => {
    onFilter(filterId);
    onClose();
  };

  // Handle keyboard navigation
  const handleKeyDown = (e) => {
    if (e.key === 'Escape') {
      onClose();
    } else if (e.key === 'Enter' && query.trim()) {
      handleSearch();
    }
  };

  // Click outside to close
  useEffect(() => {
    const handleClickOutside = (event) => {
      if (overlayRef.current && !overlayRef.current.contains(event.target)) {
        onClose();
      }
    };

    if (isOpen) {
      document.addEventListener('mousedown', handleClickOutside);
      return () => document.removeEventListener('mousedown', handleClickOutside);
    }
  }, [isOpen, onClose]);

  return (
    <AnimatePresence>
      {isOpen && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="fixed inset-0 z-50 bg-black/20 backdrop-blur-sm"
        >
          <motion.div
            ref={overlayRef}
            initial={{ opacity: 0, scale: 0.95, y: -20 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.95, y: -20 }}
            transition={{ type: 'spring', damping: 25, stiffness: 300 }}
            className="absolute top-20 left-1/2 transform -translate-x-1/2 w-full max-w-2xl mx-4"
          >
            <Card className="bg-white/95 backdrop-blur-md shadow-2xl border-0 overflow-hidden">
              {/* Header with search input */}
              <div className="p-6 border-b border-gray-100">
                <div className="flex items-center gap-3">
                  <div className="relative flex-1">
                    <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-5 h-5" />
                    <Input
                      ref={inputRef}
                      value={query}
                      onChange={(e) => setQuery(e.target.value)}
                      onKeyDown={handleKeyDown}
                      placeholder="Search news, topics, or ask anything..."
                      className="pl-10 pr-4 py-3 text-lg border-0 bg-gray-50 focus:bg-white transition-colors"
                    />
                    {loading && (
                      <div className="absolute right-3 top-1/2 transform -translate-y-1/2">
                        <Sparkles className="w-5 h-5 text-blue-500 animate-pulse" />
                      </div>
                    )}
                  </div>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={onClose}
                    className="text-gray-400 hover:text-gray-600"
                  >
                    <X className="w-5 h-5" />
                  </Button>
                </div>

                {/* Section tabs */}
                <div className="flex gap-2 mt-4">
                  <Button
                    variant={activeSection === 'search' ? 'default' : 'ghost'}
                    size="sm"
                    onClick={() => setActiveSection('search')}
                    className="text-sm"
                  >
                    <Search className="w-4 h-4 mr-2" />
                    Search
                  </Button>
                  <Button
                    variant={activeSection === 'filters' ? 'default' : 'ghost'}
                    size="sm"
                    onClick={() => setActiveSection('filters')}
                    className="text-sm"
                  >
                    <Filter className="w-4 h-4 mr-2" />
                    Filters
                  </Button>
                  <Button
                    variant={activeSection === 'recent' ? 'default' : 'ghost'}
                    size="sm"
                    onClick={() => setActiveSection('recent')}
                    className="text-sm"
                  >
                    <Clock className="w-4 h-4 mr-2" />
                    Recent
                  </Button>
                </div>
              </div>

              {/* Content sections */}
              <div className="max-h-96 overflow-y-auto">
                {/* Search suggestions */}
                {activeSection === 'search' && (
                  <div className="p-4">
                    {query.trim() && suggestions.length > 0 && (
                      <div className="space-y-2">
                        <h4 className="text-sm font-medium text-gray-700 mb-3">AI Suggestions</h4>
                        {suggestions.map((suggestion, index) => (
                          <motion.button
                            key={index}
                            initial={{ opacity: 0, x: -10 }}
                            animate={{ opacity: 1, x: 0 }}
                            transition={{ delay: index * 0.05 }}
                            onClick={() => handleSearch(suggestion)}
                            className="w-full text-left p-3 rounded-lg hover:bg-gray-50 transition-colors group"
                          >
                            <div className="flex items-center justify-between">
                              <span className="text-gray-800">{suggestion}</span>
                              <ArrowRight className="w-4 h-4 text-gray-400 group-hover:text-gray-600 transition-colors" />
                            </div>
                          </motion.button>
                        ))}
                      </div>
                    )}

                    {/* Trending topics */}
                    {!query.trim() && trendingTopics.length > 0 && (
                      <div className="space-y-2">
                        <h4 className="text-sm font-medium text-gray-700 mb-3 flex items-center gap-2">
                          <TrendingUp className="w-4 h-4" />
                          Trending Now
                        </h4>
                        {trendingTopics.map((topic, index) => (
                          <motion.button
                            key={index}
                            initial={{ opacity: 0, x: -10 }}
                            animate={{ opacity: 1, x: 0 }}
                            transition={{ delay: index * 0.05 }}
                            onClick={() => handleSearch(topic)}
                            className="w-full text-left p-3 rounded-lg hover:bg-gray-50 transition-colors group"
                          >
                            <div className="flex items-center justify-between">
                              <span className="text-gray-800">{topic}</span>
                              <ArrowRight className="w-4 h-4 text-gray-400 group-hover:text-gray-600 transition-colors" />
                            </div>
                          </motion.button>
                        ))}
                      </div>
                    )}
                  </div>
                )}

                {/* Quick filters */}
                {activeSection === 'filters' && (
                  <div className="p-4">
                    <h4 className="text-sm font-medium text-gray-700 mb-3">Categories</h4>
                    <div className="grid grid-cols-2 gap-2">
                      {quickFilters.map((filter) => (
                        <motion.button
                          key={filter.id}
                          initial={{ opacity: 0, scale: 0.95 }}
                          animate={{ opacity: 1, scale: 1 }}
                          onClick={() => handleFilterSelect(filter.id)}
                          className={`p-3 rounded-lg border transition-all group ${
                            currentCategory === filter.id
                              ? 'bg-blue-50 border-blue-200 text-blue-800'
                              : 'bg-white border-gray-200 hover:border-gray-300 hover:bg-gray-50'
                          }`}
                        >
                          <div className="flex items-center gap-3">
                            <span className="text-xl">{filter.icon}</span>
                            <span className="font-medium">{filter.label}</span>
                          </div>
                        </motion.button>
                      ))}
                    </div>
                  </div>
                )}

                {/* Recent searches */}
                {activeSection === 'recent' && (
                  <div className="p-4">
                    {recentSearches.length > 0 ? (
                      <div className="space-y-2">
                        <h4 className="text-sm font-medium text-gray-700 mb-3">Recent Searches</h4>
                        {recentSearches.map((search, index) => (
                          <motion.button
                            key={index}
                            initial={{ opacity: 0, x: -10 }}
                            animate={{ opacity: 1, x: 0 }}
                            transition={{ delay: index * 0.05 }}
                            onClick={() => handleSearch(search)}
                            className="w-full text-left p-3 rounded-lg hover:bg-gray-50 transition-colors group"
                          >
                            <div className="flex items-center justify-between">
                              <div className="flex items-center gap-3">
                                <Clock className="w-4 h-4 text-gray-400" />
                                <span className="text-gray-800">{search}</span>
                              </div>
                              <ArrowRight className="w-4 h-4 text-gray-400 group-hover:text-gray-600 transition-colors" />
                            </div>
                          </motion.button>
                        ))}
                      </div>
                    ) : (
                      <div className="text-center py-8 text-gray-500">
                        <Clock className="w-8 h-8 mx-auto mb-2 text-gray-300" />
                        <p>No recent searches</p>
                      </div>
                    )}
                  </div>
                )}
              </div>

              {/* Footer with keyboard shortcuts */}
              <div className="px-4 py-3 bg-gray-50 border-t border-gray-100">
                <div className="flex items-center justify-between text-xs text-gray-500">
                  <div className="flex items-center gap-4">
                    <span>Press <Badge variant="outline" className="px-1 py-0 text-xs">Enter</Badge> to search</span>
                    <span>Press <Badge variant="outline" className="px-1 py-0 text-xs">Esc</Badge> to close</span>
                  </div>
                  <span>⌘K to open</span>
                </div>
              </div>
            </Card>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
};

export default SearchOverlay;