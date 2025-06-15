"use client";

import React, { useEffect, useState, useCallback } from "react";
import { Input } from "../components/ui/input";
import { SearchInput } from "../components/ui/SearchInput";
import { ScrollArea } from "../components/ui/scroll-area";
import { Newspaper, Filter, Loader2, AlertCircle, Youtube, XCircle } from "lucide-react";
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

  return (
    <main className="min-h-screen bg-gray-50 flex flex-col items-center p-4">
      <div className="w-full max-w-4xl mt-8">
        <div className="bg-gradient-to-r from-blue-500 to-indigo-600 rounded-xl px-6 py-4 mb-8 w-fit mx-auto">
          <h1 className="text-3xl font-bold text-white flex items-center gap-2">
            <Newspaper className="w-8 h-8 text-white" /> AI News Dashboard
          </h1>
        </div>

        {/* Wrap main content in a Card */}
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

          {/* --- Search & Filters Section --- */}
          <CardContent>
            <h3 className="text-lg font-semibold mb-4">Search & Filters</h3>
            {/* Search bar */}
            <SearchInput
              onSearch={(q) => {
                setQuery(q);
                // Clear category/tag when performing a new search
                setActiveTag("");
                setActiveCategory("");
                fetchNews(q, "", "");
              }}
              placeholder="Search news articles by keyword..."
              label=""
              className="mb-4"
            />

            {/* Tag/category chips for filtering */}
            <div className="mb-2 flex items-center gap-2 text-sm text-gray-600">
              <Filter className="w-4 h-4" /> Filters:
            </div>
            <div className="flex flex-wrap gap-2 mb-6 items-center">
              {allTags.map((tag) => (
                <TagChip key={tag} label={tag} active={activeTag === tag} onClick={() => handleTagClick(tag)} />
              ))}
              {activeTag && (
                <Button
                  variant="secondary"
                  size="sm"
                  className="text-xs h-auto py-1 px-2"
                  onClick={() => handleTagClick("")} // Clear filter
                >
                  <XCircle className="w-3 h-3 mr-1" /> Clear Filter
                </Button>
              )}
            </div>
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
      </div>
    </main>
  );
};

export default NewsDashboard;
