import React, { useEffect, useState } from "react";
import { Search, Command } from "lucide-react";
import { Button } from "./ui/button";
import { Badge } from "./ui/badge";
import SearchOverlay from "./SearchOverlay";

/**
 * Search Trigger Component
 * 
 * Provides multiple ways to trigger the Arc-style search overlay:
 * - Click on search button
 * - Keyboard shortcut (Cmd/Ctrl + K)
 * - Global search hotkey
 * 
 * @param {Object} props
 * @param {Function} props.onSearch - Callback when search is performed
 * @param {Function} props.onFilter - Callback when filter is applied
 * @param {string} props.variant - Button variant ('button', 'input', 'minimal')
 * @param {string} props.className - Additional CSS classes
 * @param {Array} props.recentSearches - Recent search terms
 * @param {Array} props.trendingTopics - Trending topics
 */
export default function SearchTrigger({ 
  onSearch, 
  onFilter,
  variant = "button",
  className = "",
  recentSearches = [],
  trendingTopics = []
}) {
  const [isOverlayOpen, setIsOverlayOpen] = useState(false);
  const [isMac, setIsMac] = useState(false);

  // Detect operating system for keyboard shortcuts
  useEffect(() => {
    setIsMac(navigator.platform.toUpperCase().indexOf('MAC') >= 0);
  }, []);

  // Global keyboard shortcut handler
  useEffect(() => {
    const handleKeyDown = (event) => {
      // Cmd+K (Mac) or Ctrl+K (Windows/Linux)
      if ((event.metaKey || event.ctrlKey) && event.key === 'k') {
        event.preventDefault();
        setIsOverlayOpen(true);
      }
      
      // Alternative: Cmd+/ or Ctrl+/
      if ((event.metaKey || event.ctrlKey) && event.key === '/') {
        event.preventDefault();
        setIsOverlayOpen(true);
      }
    };

    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, []);

  const handleOpenOverlay = () => {
    setIsOverlayOpen(true);
  };

  const handleCloseOverlay = () => {
    setIsOverlayOpen(false);
  };

  const handleSearch = (query) => {
    onSearch?.(query);
    setIsOverlayOpen(false);
  };

  const handleFilter = (filter) => {
    onFilter?.(filter);
    setIsOverlayOpen(false);
  };

  // Render different variants
  const renderTrigger = () => {
    switch (variant) {
      case "input":
        return (
          <div 
            className={`relative cursor-pointer ${className}`}
            onClick={handleOpenOverlay}
          >
            <div className="flex items-center w-full px-4 py-3 bg-gray-50 border border-gray-200 rounded-xl hover:bg-gray-100 transition-colors">
              <Search size={18} className="text-gray-400 mr-3" />
              <span className="text-gray-500 flex-1">Search news with AI...</span>
              <div className="flex items-center gap-1 text-xs text-gray-400">
                <Badge variant="outline" className="px-2 py-1 text-xs border-gray-300">
                  {isMac ? '⌘' : 'Ctrl'} K
                </Badge>
              </div>
            </div>
          </div>
        );
      
      case "minimal":
        return (
          <Button
            variant="ghost"
            size="icon"
            onClick={handleOpenOverlay}
            className={`h-10 w-10 rounded-full hover:bg-gray-100 ${className}`}
            aria-label="Open search"
          >
            <Search size={18} />
          </Button>
        );
      
      case "button":
      default:
        return (
          <Button
            variant="outline"
            onClick={handleOpenOverlay}
            className={`flex items-center gap-2 ${className}`}
          >
            <Search size={16} />
            <span>Search</span>
            <div className="hidden sm:flex items-center gap-1 ml-2">
              <Badge variant="secondary" className="px-2 py-1 text-xs">
                {isMac ? '⌘' : 'Ctrl'} K
              </Badge>
            </div>
          </Button>
        );
    }
  };

  return (
    <>
      {renderTrigger()}
      
      <SearchOverlay
        isOpen={isOverlayOpen}
        onClose={handleCloseOverlay}
        onSearch={handleSearch}
        onFilter={handleFilter}
        recentSearches={recentSearches}
        trendingTopics={trendingTopics}
      />
    </>
  );
}

/**
 * Hook for managing search state and recent searches
 */
export function useSearchState() {
  const [recentSearches, setRecentSearches] = useState([]);
  const [trendingTopics] = useState([
    "AI Technology",
    "Climate Change",
    "Cryptocurrency",
    "Space Exploration",
    "Renewable Energy",
    "Quantum Computing"
  ]);

  // Load recent searches from localStorage on mount
  useEffect(() => {
    try {
      const stored = localStorage.getItem('ai-news-recent-searches');
      if (stored) {
        setRecentSearches(JSON.parse(stored));
      }
    } catch (error) {
      console.error('Failed to load recent searches:', error);
    }
  }, []);

  // Add a new search to recent searches
  const addRecentSearch = (query) => {
    if (!query || typeof query !== 'string') return;
    
    const trimmedQuery = query.trim();
    if (trimmedQuery.length === 0) return;

    setRecentSearches(prev => {
      // Remove if already exists
      const filtered = prev.filter(search => search !== trimmedQuery);
      // Add to beginning
      const updated = [trimmedQuery, ...filtered].slice(0, 10); // Keep only 10 recent searches
      
      // Save to localStorage
      try {
        localStorage.setItem('ai-news-recent-searches', JSON.stringify(updated));
      } catch (error) {
        console.error('Failed to save recent searches:', error);
      }
      
      return updated;
    });
  };

  // Clear recent searches
  const clearRecentSearches = () => {
    setRecentSearches([]);
    try {
      localStorage.removeItem('ai-news-recent-searches');
    } catch (error) {
      console.error('Failed to clear recent searches:', error);
    }
  };

  return {
    recentSearches,
    trendingTopics,
    addRecentSearch,
    clearRecentSearches
  };
}