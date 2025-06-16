import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Search, X, Clock, TrendingUp, Users, Zap, Mic, Filter, ArrowRight } from 'lucide-react';
import { useContextAware } from '../contexts/ContextProvider';

const ArcSearchOverlay = ({ isOpen, onClose, onSearch }) => {
  const [query, setQuery] = useState('');
  const [suggestions, setSuggestions] = useState([]);
  const [activeFilter, setActiveFilter] = useState('all');
  const [isVoiceActive, setIsVoiceActive] = useState(false);
  const [recentSearches, setRecentSearches] = useState([]);
  const [trendingTopics, setTrendingTopics] = useState([]);
  const [personalizedFilters, setPersonalizedFilters] = useState([]);
  
  const searchInputRef = useRef(null);
  const overlayRef = useRef(null);
  
  const {
    state,
    dispatch,
    trackInteraction,
    getPersonalizedPrompts,
    getSuggestedSkills,
    shouldShowFeature
  } = useContextAware();

  // Focus search input when overlay opens
  useEffect(() => {
    if (isOpen && searchInputRef.current) {
      searchInputRef.current.focus();
    }
  }, [isOpen]);

  // Load contextual data when overlay opens
  useEffect(() => {
    if (isOpen) {
      loadContextualData();
      trackInteraction('search_overlay_open', 'arc_search');
    }
  }, [isOpen]);

  // Load contextual search data
  const loadContextualData = useCallback(() => {
    // Recent searches from context
    setRecentSearches(state.session.searchHistory.slice(-5));
    
    // Trending topics based on user preferences and time
    const trending = [
      'AI Ethics in 2024',
      'Machine Learning Breakthroughs',
      'OpenAI GPT-5 Rumors',
      'AI Regulation Updates',
      'Autonomous Vehicles'
    ];
    setTrendingTopics(trending);
    
    // Personalized filters based on user behavior
    const filters = ['all'];
    if (state.user.behavior.articlesRead > 3) filters.push('in-depth');
    if (state.session.readArticles.length > 0) filters.push('related');
    if (state.environment.timeOfDay === 'morning') filters.push('breaking');
    if (shouldShowFeature('social-recommendations')) filters.push('trending');
    
    setPersonalizedFilters(filters);
  }, [state, shouldShowFeature]);

  // Handle search input changes with real-time suggestions
  const handleInputChange = useCallback((e) => {
    const value = e.target.value;
    setQuery(value);
    
    if (value.length > 2) {
      // Generate contextual suggestions
      const contextualSuggestions = generateSuggestions(value);
      setSuggestions(contextualSuggestions);
    } else {
      setSuggestions([]);
    }
    
    trackInteraction('search_input', 'typing', { query: value });
  }, [trackInteraction]);

  // Generate contextual suggestions
  const generateSuggestions = useCallback((searchQuery) => {
    const suggestions = [];
    
    // AI-powered suggestions based on context
    const aiSuggestions = [
      `${searchQuery} latest developments`,
      `${searchQuery} impact on industry`,
      `${searchQuery} expert opinions`,
      `${searchQuery} vs alternatives`
    ];
    
    // Add contextual suggestions based on user behavior
    if (state.user.behavior.searchQueries.length > 0) {
      const relatedQueries = state.user.behavior.searchQueries
        .filter(q => q.toLowerCase().includes(searchQuery.toLowerCase()))
        .slice(0, 2);
      suggestions.push(...relatedQueries.map(q => ({ text: q, type: 'recent' })));
    }
    
    // Add AI suggestions
    suggestions.push(...aiSuggestions.map(s => ({ text: s, type: 'ai' })));
    
    // Add trending suggestions if relevant
    const relevantTrending = trendingTopics
      .filter(topic => topic.toLowerCase().includes(searchQuery.toLowerCase()))
      .slice(0, 2);
    suggestions.push(...relevantTrending.map(t => ({ text: t, type: 'trending' })));
    
    return suggestions.slice(0, 6);
  }, [state.user.behavior.searchQueries, trendingTopics]);

  // Handle search execution
  const executeSearch = useCallback((searchQuery = query) => {
    if (!searchQuery.trim()) return;
    
    // Track search in context
    dispatch({
      type: 'ADD_SEARCH',
      payload: { query: searchQuery, results: 0 }
    });
    
    trackInteraction('search_execute', 'arc_search', {
      query: searchQuery,
      filter: activeFilter,
      source: 'overlay'
    });
    
    // Execute search with context-aware parameters
    onSearch({
      query: searchQuery,
      filter: activeFilter,
      context: {
        userPreferences: state.user.preferences,
        timeOfDay: state.environment.timeOfDay,
        deviceType: state.environment.deviceType,
        recentTopics: state.user.behavior.searchQueries.slice(-3)
      }
    });
    
    onClose();
  }, [query, activeFilter, dispatch, trackInteraction, onSearch, onClose, state]);

  // Voice search functionality
  const toggleVoiceSearch = useCallback(() => {
    if (!('webkitSpeechRecognition' in window)) {
      alert('Voice search not supported in this browser');
      return;
    }
    
    if (isVoiceActive) {
      setIsVoiceActive(false);
      return;
    }
    
    const recognition = new window.webkitSpeechRecognition();
    recognition.continuous = false;
    recognition.interimResults = false;
    recognition.lang = 'en-US';
    
    recognition.onstart = () => {
      setIsVoiceActive(true);
      trackInteraction('voice_search_start', 'arc_search');
    };
    
    recognition.onresult = (event) => {
      const transcript = event.results[0][0].transcript;
      setQuery(transcript);
      setIsVoiceActive(false);
      executeSearch(transcript);
    };
    
    recognition.onerror = () => {
      setIsVoiceActive(false);
    };
    
    recognition.onend = () => {
      setIsVoiceActive(false);
    };
    
    recognition.start();
  }, [isVoiceActive, executeSearch, trackInteraction]);

  // Handle keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (!isOpen) return;
      
      if (e.key === 'Escape') {
        onClose();
      } else if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        executeSearch();
      } else if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
        e.preventDefault();
        // Handle suggestion navigation (simplified)
      }
    };
    
    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, onClose, executeSearch]);

  // Handle click outside to close
  useEffect(() => {
    const handleClickOutside = (e) => {
      if (overlayRef.current && !overlayRef.current.contains(e.target)) {
        onClose();
      }
    };
    
    if (isOpen) {
      document.addEventListener('mousedown', handleClickOutside);
    }
    
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-start justify-center pt-20">
      <div 
        ref={overlayRef}
        className="bg-white dark:bg-gray-900 rounded-2xl shadow-2xl w-full max-w-2xl mx-4 overflow-hidden animate-in slide-in-from-top-4 duration-300"
      >
        {/* Header */}
        <div className="p-6 border-b border-gray-200 dark:border-gray-700">
          <div className="flex items-center gap-4">
            <div className="relative flex-1">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-5 h-5" />
              <input
                ref={searchInputRef}
                type="text"
                value={query}
                onChange={handleInputChange}
                placeholder={`Search AI news... ${state.environment.timeOfDay === 'morning' ? '☀️' : state.environment.timeOfDay === 'evening' ? '🌙' : ''}`}
                className="w-full pl-10 pr-20 py-3 text-lg border-0 focus:ring-0 bg-gray-50 dark:bg-gray-800 rounded-xl"
              />
              <div className="absolute right-3 top-1/2 transform -translate-y-1/2 flex items-center gap-2">
                {shouldShowFeature('voice-search') && (
                  <button
                    onClick={toggleVoiceSearch}
                    className={`p-2 rounded-lg transition-colors ${
                      isVoiceActive 
                        ? 'bg-red-100 text-red-600 dark:bg-red-900 dark:text-red-400' 
                        : 'hover:bg-gray-200 dark:hover:bg-gray-700 text-gray-500'
                    }`}
                  >
                    <Mic className="w-4 h-4" />
                  </button>
                )}
                <button
                  onClick={onClose}
                  className="p-2 hover:bg-gray-200 dark:hover:bg-gray-700 rounded-lg transition-colors"
                >
                  <X className="w-4 h-4" />
                </button>
              </div>
            </div>
          </div>
          
          {/* Contextual Filters */}
          <div className="flex items-center gap-2 mt-4">
            <Filter className="w-4 h-4 text-gray-500" />
            <div className="flex gap-2 flex-wrap">
              {personalizedFilters.map((filter) => (
                <button
                  key={filter}
                  onClick={() => setActiveFilter(filter)}
                  className={`px-3 py-1 rounded-full text-sm transition-colors ${
                    activeFilter === filter
                      ? 'bg-blue-100 text-blue-700 dark:bg-blue-900 dark:text-blue-300'
                      : 'bg-gray-100 text-gray-600 hover:bg-gray-200 dark:bg-gray-800 dark:text-gray-400 dark:hover:bg-gray-700'
                  }`}
                >
                  {filter.charAt(0).toUpperCase() + filter.slice(1)}
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Content */}
        <div className="max-h-96 overflow-y-auto">
          {/* Suggestions */}
          {suggestions.length > 0 && (
            <div className="p-4 border-b border-gray-200 dark:border-gray-700">
              <h3 className="text-sm font-medium text-gray-500 dark:text-gray-400 mb-3">Suggestions</h3>
              <div className="space-y-2">
                {suggestions.map((suggestion, index) => (
                  <button
                    key={index}
                    onClick={() => executeSearch(suggestion.text)}
                    className="w-full flex items-center gap-3 p-3 hover:bg-gray-50 dark:hover:bg-gray-800 rounded-lg transition-colors text-left"
                  >
                    <div className={`p-1 rounded ${
                      suggestion.type === 'ai' ? 'bg-purple-100 text-purple-600 dark:bg-purple-900 dark:text-purple-400' :
                      suggestion.type === 'trending' ? 'bg-orange-100 text-orange-600 dark:bg-orange-900 dark:text-orange-400' :
                      'bg-gray-100 text-gray-600 dark:bg-gray-800 dark:text-gray-400'
                    }`}>
                      {suggestion.type === 'ai' ? <Zap className="w-3 h-3" /> :
                       suggestion.type === 'trending' ? <TrendingUp className="w-3 h-3" /> :
                       <Clock className="w-3 h-3" />}
                    </div>
                    <span className="flex-1">{suggestion.text}</span>
                    <ArrowRight className="w-4 h-4 text-gray-400" />
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Recent Searches */}
          {recentSearches.length > 0 && query.length === 0 && (
            <div className="p-4 border-b border-gray-200 dark:border-gray-700">
              <h3 className="text-sm font-medium text-gray-500 dark:text-gray-400 mb-3">Recent Searches</h3>
              <div className="space-y-2">
                {recentSearches.map((search, index) => (
                  <button
                    key={index}
                    onClick={() => executeSearch(search.query)}
                    className="w-full flex items-center gap-3 p-3 hover:bg-gray-50 dark:hover:bg-gray-800 rounded-lg transition-colors text-left"
                  >
                    <Clock className="w-4 h-4 text-gray-400" />
                    <span className="flex-1">{search.query}</span>
                    <span className="text-xs text-gray-400">
                      {search.results} results
                    </span>
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Trending Topics */}
          {query.length === 0 && (
            <div className="p-4">
              <h3 className="text-sm font-medium text-gray-500 dark:text-gray-400 mb-3">Trending in AI</h3>
              <div className="grid grid-cols-1 gap-2">
                {trendingTopics.map((topic, index) => (
                  <button
                    key={index}
                    onClick={() => executeSearch(topic)}
                    className="flex items-center gap-3 p-3 hover:bg-gray-50 dark:hover:bg-gray-800 rounded-lg transition-colors text-left"
                  >
                    <TrendingUp className="w-4 h-4 text-orange-500" />
                    <span className="flex-1">{topic}</span>
                    <ArrowRight className="w-4 h-4 text-gray-400" />
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Personalized Prompts */}
          {query.length === 0 && (
            <div className="p-4 bg-gradient-to-r from-blue-50 to-purple-50 dark:from-blue-900/20 dark:to-purple-900/20">
              <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-3">AI Assistant Suggestions</h3>
              <div className="space-y-2">
                {getPersonalizedPrompts().slice(0, 2).map((prompt, index) => (
                  <div key={index} className="flex items-center gap-3 p-3 bg-white/50 dark:bg-gray-800/50 rounded-lg">
                    <Zap className="w-4 h-4 text-purple-500" />
                    <span className="text-sm">{prompt}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="p-4 bg-gray-50 dark:bg-gray-800 border-t border-gray-200 dark:border-gray-700">
          <div className="flex items-center justify-between text-xs text-gray-500 dark:text-gray-400">
            <div className="flex items-center gap-4">
              <span>Press Enter to search</span>
              <span>Esc to close</span>
              {shouldShowFeature('voice-search') && <span>Click mic for voice</span>}
            </div>
            <div className="flex items-center gap-2">
              <span>Powered by AI</span>
              <Zap className="w-3 h-3" />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ArcSearchOverlay;