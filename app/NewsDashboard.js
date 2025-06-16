"use client";

import React, { useEffect, useState, useCallback } from "react";
import { Input } from "../components/ui/input";
import { SearchInput } from "../components/ui/SearchInput";
import { ScrollArea } from "../components/ui/scroll-area";
import { Newspaper, Filter, Loader2, AlertCircle, Youtube, XCircle, Search, Users, Bell, User } from "lucide-react";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "../components/ui/dropdown-menu";
import { Button } from "../components/ui/button";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "../components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "../components/ui/tabs";
import { Badge } from "../components/ui/badge";
import { Separator } from "../components/ui/separator";
import { motion } from "framer-motion";
import NewsCard from "../components/NewsCard";
import TagChip from "../components/TagChip";
import { Popover, PopoverTrigger, PopoverContent } from "../components/ui/popover";
import SearchTrigger, { useSearchState } from "../components/SearchTrigger";

// Social Features Components
import SocialRecommendations from "../components/SocialRecommendations";
import UserProfile from "../components/UserProfile";
import NotificationCenter from "../components/NotificationCenter";
import GroupManager from "../components/GroupManager";

// Context-Aware Components
import { ContextProvider, useContextAware } from "../contexts/ContextProvider";
import ArcSearchOverlay from "../components/ArcSearchOverlay";
import SkillOrchestrator from "../components/SkillOrchestrator";
import ExperimentationEngine from "../components/ExperimentationEngine";
import MonitoringDashboard from "../components/MonitoringDashboard";

/**
 * @typedef {Object} NewsItem
 * @property {string} id
 * @property {string} title
 * @property {string} description
 * @property {string} content
 * @property {string} url
 * @property {string} image
 * @property {string} publishedAt
 * @property {{name: string, url: string}} source
 * @property {string} category
 * @property {string[]} tags
 */

// NewsDashboard: Main component for the AI News Dashboard
// Fetches news from the API, supports search, filtering, and summarization
const NewsDashboard = () => {
  // State for search query, news articles, loading, etc.
  const [query, setQuery] = useState("");
  const [news, setNews] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [summaries, setSummaries] = useState({}); // {id: summary}
  const [activeTag, setActiveTag] = useState("");
  const [allTags, setAllTags] = useState([]);
  const [activeCategory, setActiveCategory] = useState("general"); // Default category

  // --- YouTube Summarizer State ---
  const [ytUrl, setYtUrl] = useState("");
  const [ytSummary, setYtSummary] = useState("");
  const [ytLoading, setYtLoading] = useState(false);
  const [ytError, setYtError] = useState("");
  const [ytPopoverOpen, setYtPopoverOpen] = useState(false);

  // Summarization engine state
  const [modelEngine, setModelEngine] = useState("o4");

  // Arc-style search state
  const { recentSearches, trendingTopics, addRecentSearch } = useSearchState();

  // Social Features State
  const [activeView, setActiveView] = useState('news'); // 'news', 'social', 'profile', 'groups'
  const [currentUserId] = useState('demo-user-1'); // In real app, get from auth
  const [showNotifications, setShowNotifications] = useState(false);
  const [unreadNotifications, setUnreadNotifications] = useState(3); // Mock count

  // Context-Aware Features State
  const [showArcSearch, setShowArcSearch] = useState(false);
  const [showSkillOrchestrator, setShowSkillOrchestrator] = useState(false);
  const [showExperimentation, setShowExperimentation] = useState(false);
  const [showMonitoring, setShowMonitoring] = useState(false);
  const [systemHealth, setSystemHealth] = useState(null);
  const [contextAwareEnabled, setContextAwareEnabled] = useState(true);

  // Fetch news from the API
  const fetchNews = useCallback(async (searchQuery = "", tag = "", category = "") => {
    setLoading(true);
    setError("");
    try {
      // Prioritize category if provided, otherwise use search query
      const effectiveQuery = category ? '' : searchQuery;
      const effectiveCategory = category ? category : '';

      let url = `/api/news?q=${encodeURIComponent(effectiveQuery)}`;
      if (effectiveCategory) url += `&category=${encodeURIComponent(effectiveCategory)}`;
      if (tag) url += `&tag=${encodeURIComponent(tag)}`;

      const res = await fetch(url);
      const data = await res.json();
      if (data.error) {
        setError(data.error);
        setNews([]);
      } else {
        setNews(data.articles || []);
        // Collect all unique tags for filtering
        const tagsSet = new Set();
        for (const article of (data.articles || [])) {
          for (const t of (article.tags || [])) {
            tagsSet.add(t);
          }
        }
        setAllTags(Array.from(tagsSet).sort());
      }
    } catch (err) {
      setError("Failed to fetch news.");
      setNews([]);
    }
    setLoading(false);
  }, []);

  // Fetch news on mount and when activeTag or activeCategory changes
  useEffect(() => {
    // Don't trigger initial fetch here if category is handling it
    if (activeCategory) {
      fetchNews("", "", activeCategory); // Fetch based on category
    } else if (activeTag) {
      fetchNews("", activeTag, ""); // Fetch based on tag (if no category)
    } else {
      fetchNews("", "", "general"); // Default fetch for general category
    }
  }, [activeTag, activeCategory, fetchNews]);

  // Handle tag click
  const handleTagClick = (tag) => {
    // Clear category when a tag is clicked
    setActiveCategory("");
    setActiveTag(tag === activeTag ? "" : tag);
  };

  // Handle category change from Tabs
  const handleCategoryChange = (category) => {
    setActiveCategory(category);
    setActiveTag(""); // Clear active tag when category changes
    setQuery(""); // Optionally clear search query too
    // Fetching is handled by the useEffect dependency on activeCategory
  };

  // Call o4-mini-high model for summarization
  const summarizeArticle = async (id, content) => {
    if (!content) return;
    setSummaries((prev) => ({ ...prev, [id]: "Summarizing..." }));
    try {
      // Call API route based on selected summarization engine
      const apiRoute = modelEngine === "openai" ? "/api/summarize-openai" : "/api/summarize";
      const res = await fetch(apiRoute, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ content }),
      });
      const data = await res.json();
      setSummaries((prev) => ({ ...prev, [id]: data.summary || "" }));
    } catch (err) {
      setSummaries((prev) => ({ ...prev, [id]: "Error summarizing." }));
    }
  };

  // --- Arc-style Search Handlers ---
  const handleArcSearch = (searchQuery) => {
    setQuery(searchQuery);
    setActiveCategory(""); // Clear category when searching
    setActiveTag(""); // Clear tag when searching
    addRecentSearch(searchQuery); // Add to recent searches
    fetchNews(searchQuery, "", ""); // Perform search
  };

  const handleArcFilter = (filterValue) => {
    setActiveCategory(filterValue);
    setActiveTag(""); // Clear tag when filtering by category
    setQuery(""); // Clear search query
    fetchNews("", "", filterValue); // Filter by category
  };

  // --- YouTube Summarizer Logic ---
  const handleYtSummarize = async (e) => {
    e.preventDefault();
    setYtSummary("");
    setYtError("");
    setYtLoading(true);
    setYtPopoverOpen(true);
    try {
      const res = await fetch("/api/summarize-youtube", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url: ytUrl, engine: modelEngine }),
      });
      const data = await res.json();
      setYtSummary(data.summary || "");
      if (!data.summary) setYtError("Could not summarize video.");
    } catch (err) {
      setYtError("Failed to summarize YouTube video.");
    }
    setYtLoading(false);
  };

  // Context-Aware Effects
  useEffect(() => {
    // Keyboard shortcuts
    const handleKeyDown = (e) => {
      if (e.ctrlKey || e.metaKey) {
        switch (e.key) {
          case 'k':
            e.preventDefault();
            setShowArcSearch(true);
            break;
          case 'j':
            e.preventDefault();
            setShowSkillOrchestrator(true);
            break;
          case 'm':
            e.preventDefault();
            setShowMonitoring(true);
            break;
          case 'e':
            e.preventDefault();
            setShowExperimentation(true);
            break;
        }
      }
      if (e.key === 'Escape') {
        setShowArcSearch(false);
        setShowSkillOrchestrator(false);
        setShowMonitoring(false);
        setShowExperimentation(false);
      }
    };

    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, []);

  // Health check effect
  useEffect(() => {
    const checkSystemHealth = async () => {
      try {
        const response = await fetch('/api/health');
        const health = await response.json();
        setSystemHealth(health);
      } catch (error) {
        console.error('Health check failed:', error);
        setSystemHealth({ status: 'unhealthy', message: 'Health check failed' });
      }
    };

    checkSystemHealth();
    const interval = setInterval(checkSystemHealth, 30000); // Check every 30 seconds
    return () => clearInterval(interval);
  }, []);

  return (
    <ContextProvider>
      <main className="min-h-screen bg-gray-50 flex flex-col items-center p-4">
      <div className="w-full max-w-6xl mt-8">
        {/* Header with Navigation */}
        <div className="bg-gradient-to-r from-blue-500 to-indigo-600 rounded-xl px-6 py-4 mb-8">
          <div className="flex items-center justify-between">
            <h1 className="text-3xl font-bold text-white flex items-center gap-2">
              <Newspaper className="w-8 h-8 text-white" /> AI News Dashboard
            </h1>
            
            {/* Navigation and User Actions */}
            <div className="flex items-center space-x-4">
              {/* View Toggle */}
              <div className="flex bg-white/20 rounded-lg p-1">
                <button
                  onClick={() => setActiveView('news')}
                  className={`px-3 py-1.5 text-sm font-medium rounded-md transition-colors ${
                    activeView === 'news' ? 'bg-white text-blue-600' : 'text-white hover:bg-white/20'
                  }`}
                >
                  News
                </button>
                <button
                  onClick={() => setActiveView('social')}
                  className={`px-3 py-1.5 text-sm font-medium rounded-md transition-colors ${
                    activeView === 'social' ? 'bg-white text-blue-600' : 'text-white hover:bg-white/20'
                  }`}
                >
                  Social
                </button>
                <button
                  onClick={() => setActiveView('groups')}
                  className={`px-3 py-1.5 text-sm font-medium rounded-md transition-colors ${
                    activeView === 'groups' ? 'bg-white text-blue-600' : 'text-white hover:bg-white/20'
                  }`}
                >
                  Groups
                </button>
                <button
                  onClick={() => setActiveView('profile')}
                  className={`px-3 py-1.5 text-sm font-medium rounded-md transition-colors ${
                    activeView === 'profile' ? 'bg-white text-blue-600' : 'text-white hover:bg-white/20'
                  }`}
                >
                  Profile
                </button>
              </div>
              
              {/* Context-Aware Features */}
              {contextAwareEnabled && (
                <div className="flex items-center space-x-2">
                  {/* Arc Search */}
                  <button
                    onClick={() => setShowArcSearch(true)}
                    className="p-2 text-white hover:bg-white/20 rounded-lg transition-colors"
                    title="Arc-style Search (Ctrl+K)"
                  >
                    <Search className="w-5 h-5" />
                  </button>
                  
                  {/* AI Skills */}
                  <button
                    onClick={() => setShowSkillOrchestrator(true)}
                    className="p-2 text-white hover:bg-white/20 rounded-lg transition-colors"
                    title="AI Skills Orchestrator"
                  >
                    <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
                      <path d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                  </button>
                  
                  {/* Experimentation */}
                  <button
                    onClick={() => setShowExperimentation(true)}
                    className="p-2 text-white hover:bg-white/20 rounded-lg transition-colors"
                    title="A/B Testing & Experiments"
                  >
                    <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
                      <path d="M9 2a1 1 0 000 2h2a1 1 0 100-2H9z" />
                      <path fillRule="evenodd" d="M4 5a2 2 0 012-2v1a1 1 0 001 1h6a1 1 0 001-1V3a2 2 0 012 2v6a2 2 0 01-2 2H6a2 2 0 01-2-2V5zm2.5 7a1.5 1.5 0 100-3 1.5 1.5 0 000 3zm2.45.5a2.5 2.5 0 11-3.4-3.4l1.78-1.77a.75.75 0 011.06 1.06L6.95 9.33a2.5 2.5 0 013.4 3.4l1.77 1.78a.75.75 0 11-1.06 1.06L9.28 13.5z" clipRule="evenodd" />
                    </svg>
                  </button>
                  
                  {/* System Monitoring */}
                  <button
                    onClick={() => setShowMonitoring(true)}
                    className="p-2 text-white hover:bg-white/20 rounded-lg transition-colors"
                    title="System Health & Monitoring"
                  >
                    <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
                      <path fillRule="evenodd" d="M3 3a1 1 0 000 2v8a2 2 0 002 2h2.586l-1.293 1.293a1 1 0 101.414 1.414L10 15.414l2.293 2.293a1 1 0 001.414-1.414L12.414 15H15a2 2 0 002-2V5a1 1 0 100-2H3zm11.707 4.707a1 1 0 00-1.414-1.414L10 9.586 8.707 8.293a1 1 0 00-1.414 0l-2 2a1 1 0 101.414 1.414L8 10.414l1.293 1.293a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                    </svg>
                  </button>
                </div>
              )}
              
              {/* Notifications */}
              <div className="relative">
                <button
                  onClick={() => setShowNotifications(!showNotifications)}
                  className="relative p-2 text-white hover:bg-white/20 rounded-lg transition-colors"
                >
                  <Bell className="w-5 h-5" />
                  {unreadNotifications > 0 && (
                    <span className="absolute -top-1 -right-1 bg-red-500 text-white text-xs rounded-full w-5 h-5 flex items-center justify-center">
                      {unreadNotifications}
                    </span>
                  )}
                </button>
                
                {/* Notification Dropdown */}
                {showNotifications && (
                  <div className="absolute right-0 top-full mt-2 w-80 z-50">
                    <NotificationCenter 
                      userId={currentUserId}
                      onClose={() => setShowNotifications(false)}
                      onNotificationRead={() => setUnreadNotifications(prev => Math.max(0, prev - 1))}
                    />
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>

        {/* Main Content Area */}
        {activeView === 'news' && (
          <Card className="w-full shadow-lg bg-white rounded-lg">

            {/* --- YouTube Summarizer Section --- */}
            <Card className="border-0 rounded-none shadow-none">
              <CardHeader>
                <CardTitle className="flex items-center gap-2"><Youtube className="w-5 h-5 text-red-600" /> YouTube News Bite</CardTitle>
                <CardDescription>Paste a YouTube URL to get an AI-generated news summary.</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="flex justify-end mb-4">
                  <Button variant="outline" size="sm" onClick={() => setYtUrl('https://www.youtube.com/watch?v=dQw4w9WgXcQ')}>Use Sample URL</Button>
                </div>
                <Popover open={ytPopoverOpen} onOpenChange={setYtPopoverOpen}>
                  <form onSubmit={handleYtSummarize} className="flex flex-col sm:flex-row gap-2">
                    <Input value={ytUrl} onChange={(e) => setYtUrl(e.target.value)} placeholder="YouTube URL" />
                    <PopoverTrigger asChild>
                      <Button type="submit">Summarize</Button>
                    </PopoverTrigger>
                  </form>
                  <PopoverContent className="w-80 p-4">
                    <div className="flex items-center gap-2 mb-2">
                      <span className="text-sm font-medium">Engine:</span>
                      <DropdownMenu>
                        <DropdownMenuTrigger asChild>
                          <Button variant="outline" size="xs">{modelEngine === "openai" ? "OpenAI" : "O4"}</Button>
                        </DropdownMenuTrigger>
                        <DropdownMenuContent>
                          <DropdownMenuItem onSelect={() => setModelEngine("o4")}>O4 Model</DropdownMenuItem>
                          <DropdownMenuItem onSelect={() => setModelEngine("openai")}>OpenAI</DropdownMenuItem>
                        </DropdownMenuContent>
                      </DropdownMenu>
                    </div>
                    <div className="max-h-40 overflow-auto">
                      {ytLoading ? <Loader2 className="h-6 w-6 animate-spin" /> : ytError ? <span>{ytError}</span> : <div className="whitespace-pre-wrap text-sm">{ytSummary}</div>}
                    </div>
                  </PopoverContent>
                </Popover>
                {ytError && (
                  <Card className="mt-4 p-4 bg-red-50 border-red-200">
                    <div className="flex items-center text-red-700">
                      <AlertCircle className="w-5 h-5 mr-2" />
                      <span>{ytError}</span>
                    </div>
                  </Card>
                )}
              </CardContent>
            </Card>

            <Separator className="my-6" />

            {/* --- Arc-Style Search Section --- */}
            <CardContent>
              <div className="flex items-center justify-between mb-6">
                <h3 className="text-lg font-semibold">Discover News</h3>
                <SearchTrigger 
                  onSearch={handleArcSearch}
                  onFilter={handleArcFilter}
                  recentSearches={recentSearches}
                  trendingTopics={trendingTopics}
                  currentQuery={query}
                  currentCategory={activeCategory}
                />
              </div>

              {/* Active filters display */}
              {(query || activeCategory || activeTag) && (
                <div className="mb-4 p-3 bg-blue-50 rounded-lg border border-blue-200">
                  <div className="flex items-center gap-2 text-sm text-blue-700 mb-2">
                    <Filter className="w-4 h-4" /> Active Filters:
                  </div>
                  <div className="flex flex-wrap gap-2">
                    {query && (
                      <Badge variant="secondary" className="bg-blue-100 text-blue-800">
                        Search: {query}
                        <button 
                          onClick={() => {
                            setQuery("");
                            fetchNews("", activeTag, activeCategory);
                          }}
                          className="ml-2 hover:text-blue-600"
                        >
                          <XCircle className="w-3 h-3" />
                        </button>
                      </Badge>
                    )}
                    {activeCategory && (
                      <Badge variant="secondary" className="bg-green-100 text-green-800">
                        Category: {activeCategory}
                        <button 
                          onClick={() => {
                            setActiveCategory("");
                            fetchNews(query, activeTag, "");
                          }}
                          className="ml-2 hover:text-green-600"
                        >
                          <XCircle className="w-3 h-3" />
                        </button>
                      </Badge>
                    )}
                    {activeTag && (
                      <Badge variant="secondary" className="bg-purple-100 text-purple-800">
                        Tag: {activeTag}
                        <button 
                          onClick={() => {
                            setActiveTag("");
                            fetchNews(query, "", activeCategory);
                          }}
                          className="ml-2 hover:text-purple-600"
                        >
                          <XCircle className="w-3 h-3" />
                        </button>
                      </Badge>
                    )}
                  </div>
                </div>
              )}

              {/* Tag chips for quick filtering */}
              {allTags.length > 0 && (
                <div className="mb-6">
                  <div className="mb-2 flex items-center gap-2 text-sm text-gray-600">
                    <Filter className="w-4 h-4" /> Quick Tags:
                  </div>
                  <div className="flex flex-wrap gap-2">
                    {allTags.slice(0, 8).map((tag) => (
                      <TagChip key={tag} label={tag} active={activeTag === tag} onClick={() => handleTagClick(tag)} />
                    ))}
                  </div>
                </div>
              )}
            </CardContent>

            <Separator className="mb-6" />

            {/* --- Category Tabs --- */}
            <CardContent>
              <Tabs value={activeCategory} onValueChange={handleCategoryChange} className="w-full">
                <TabsList className="grid w-full grid-cols-2 sm:grid-cols-3 md:grid-cols-5 gap-2">
                  <TabsTrigger value="general">General</TabsTrigger>
                  <TabsTrigger value="technology">Technology</TabsTrigger>
                  <TabsTrigger value="business">Business</TabsTrigger>
                  <TabsTrigger value="sports">Sports</TabsTrigger>
                  <TabsTrigger value="world">World</TabsTrigger>
                  {/* Add more categories as needed */}
                </TabsList>
              </Tabs>
            </CardContent>

            {/* --- News Feed Section --- */}
            <CardContent className="pt-0"> {/* Adjust padding if needed */}
              <h3 className="text-lg font-semibold mb-4">News Feed{activeCategory ? ` - ${activeCategory.charAt(0).toUpperCase() + activeCategory.slice(1)}` : ''}</h3>
              {/* Loading/Error/Empty States & News Grid */}
              <div>
                {loading && (
                  <div className="flex justify-center items-center p-10">
                    <Loader2 className="h-8 w-8 animate-spin text-blue-500" />
                    <span className="ml-3 text-gray-500">Loading news...</span>
                  </div>
                )}
                {error && (
                  <Card className="mt-4 p-4 bg-red-50 border-red-200">
                    <div className="flex items-center text-red-700">
                      <AlertCircle className="w-5 h-5 mr-2" />
                      <span>{error}</span>
                    </div>
                  </Card>
                )}
                {!loading && news.length === 0 && !error && (
                  <Card className="mt-6 bg-gray-50 border-dashed border-gray-300">
                    <CardContent className="text-center text-gray-500 p-6">
                      No news articles found. Try a different search or clear filters.
                    </CardContent>
                    <CardFooter className="flex justify-center pb-6">
                      <Button variant="outline" onClick={() => fetchNews(query, activeTag, activeCategory)}>
                        Refresh
                      </Button>
                    </CardFooter>
                  </Card>
                )}
                {!loading && news.length > 0 && (
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    {news.map((article) => (
                      <NewsCard
                        key={article.id}
                        article={article}
                        onTagClick={handleTagClick}
                      />
                    ))}
                  </div>
                )}
              </div>
            </CardContent>

          </Card>
        )}

        {/* Social Feed View */}
        {activeView === 'social' && (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Main Social Feed */}
            <div className="lg:col-span-2">
              <Card className="shadow-lg border-0 bg-white/80 backdrop-blur-sm">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Users className="w-6 h-6 text-blue-500" />
                    Social Feed
                  </CardTitle>
                  <CardDescription>
                    Discover articles shared by your network and trending in the community
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <SocialRecommendations userId={currentUserId} />
                </CardContent>
              </Card>
            </div>
            
            {/* Sidebar */}
            <div className="space-y-6">
              {/* Quick Stats */}
              <Card className="bg-gradient-to-br from-blue-50 to-indigo-50 border-blue-200">
                <CardHeader className="pb-3">
                  <CardTitle className="text-lg">Your Network</CardTitle>
                </CardHeader>
                <CardContent className="space-y-3">
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-600">Following</span>
                    <span className="font-semibold text-blue-600">127</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-600">Followers</span>
                    <span className="font-semibold text-blue-600">89</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-600">Articles Shared</span>
                    <span className="font-semibold text-blue-600">23</span>
                  </div>
                </CardContent>
              </Card>
              
              {/* Trending Topics */}
              <Card>
                <CardHeader className="pb-3">
                  <CardTitle className="text-lg">Trending Topics</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-2">
                    {['AI Ethics', 'Climate Tech', 'Web3', 'Quantum Computing', 'Biotech'].map((topic, index) => (
                      <div key={topic} className="flex items-center justify-between p-2 hover:bg-gray-50 rounded-lg cursor-pointer">
                        <span className="text-sm font-medium">{topic}</span>
                        <Badge variant="secondary" className="text-xs">
                          {Math.floor(Math.random() * 100) + 20}
                        </Badge>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            </div>
          </div>
        )}

        {/* Groups View */}
        {activeView === 'groups' && (
          <Card className="w-full shadow-lg border-0 bg-white/80 backdrop-blur-sm">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Users className="w-6 h-6 text-green-500" />
                Groups & Communities
              </CardTitle>
              <CardDescription>
                Join groups, discover communities, and share knowledge with like-minded people
              </CardDescription>
            </CardHeader>
            <CardContent>
              <GroupManager userId={currentUserId} />
            </CardContent>
          </Card>
        )}

        {/* Profile View */}
        {activeView === 'profile' && (
          <Card className="w-full shadow-lg border-0 bg-white/80 backdrop-blur-sm">
            <CardContent className="p-0">
              <UserProfile userId={currentUserId} />
            </CardContent>
          </Card>
        )}
      </div>
      
      {/* Context-Aware Overlays */}
      {showArcSearch && (
        <ArcSearchOverlay
          onClose={() => setShowArcSearch(false)}
          onSearch={handleArcSearch}
          recentSearches={recentSearches}
          trendingTopics={trendingTopics}
        />
      )}
      
      {showSkillOrchestrator && (
        <SkillOrchestrator
          onClose={() => setShowSkillOrchestrator(false)}
          context={{
            currentArticles: news,
            searchQuery,
            activeCategory,
            activeTags,
            userHistory: recentSearches
          }}
        />
      )}
      
      {showExperimentation && (
        <ExperimentationEngine
          onClose={() => setShowExperimentation(false)}
          userId={currentUserId}
        />
      )}
      
      {showMonitoring && (
        <MonitoringDashboard
          onClose={() => setShowMonitoring(false)}
          systemHealth={systemHealth}
          onHealthUpdate={setSystemHealth}
        />
       )}
     </main>
    </ContextProvider>
   );
 };

export default NewsDashboard;
