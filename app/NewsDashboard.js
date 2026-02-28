"use client";

import React, { useEffect, useState, useCallback } from "react";
import { Newspaper } from "lucide-react";
import { Card, CardContent } from "../components/ui/card";
import { Separator } from "../components/ui/separator";
import YouTubeSummarizer from "./components/YouTubeSummarizer";
import NewsSearchBar from "./components/NewsSearchBar";
import CategoryTabs from "./components/CategoryTabs";
import NewsFeed from "./components/NewsFeed";

const NewsDashboard = () => {
  const [query, setQuery] = useState("");
  const [news, setNews] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [activeTag, setActiveTag] = useState("");
  const [allTags, setAllTags] = useState([]);
  const [activeCategory, setActiveCategory] = useState("general");
  const [modelEngine, setModelEngine] = useState("o4");

  const fetchNews = useCallback(async (searchQuery = "", tag = "", category = "") => {
    setLoading(true);
    setError("");
    try {
      const effectiveQuery = category ? "" : searchQuery;
      const effectiveCategory = category || "";

      let url = `/api/news?q=${encodeURIComponent(effectiveQuery)}`;
      if (effectiveCategory) url += `&category=${encodeURIComponent(effectiveCategory)}`;
      if (tag) url += `&tag=${encodeURIComponent(tag)}`;

      const res = await fetch(url);
      const data = await res.json();
      if (data.error) {
        setError(data.error);
        setNews([]);
        return;
      }

      setNews(data.articles || []);
      const tagsSet = new Set();
      for (const article of data.articles || []) {
        for (const t of article.tags || []) {
          tagsSet.add(t);
        }
      }
      setAllTags(Array.from(tagsSet).sort());
    } catch {
      setError("Failed to fetch news.");
      setNews([]);
    }
    setLoading(false);
  }, []);

  useEffect(() => {
    if (activeCategory) return fetchNews("", "", activeCategory);
    if (activeTag) return fetchNews("", activeTag, "");
    fetchNews("", "", "general");
  }, [activeTag, activeCategory, fetchNews]);

  const handleTagClick = (tag) => {
    setActiveCategory("");
    setActiveTag(tag === activeTag ? "" : tag);
  };

  const handleCategoryChange = (category) => {
    setActiveCategory(category);
    setActiveTag("");
    setQuery("");
  };

  const handleSearch = (q) => {
    setQuery(q);
    setActiveTag("");
    setActiveCategory("");
    fetchNews(q, "", "");
  };

  return (
    <main className="min-h-screen bg-gray-50 flex flex-col items-center p-4">
      <div className="w-full max-w-4xl mt-8">
        <div className="bg-gradient-to-r from-blue-500 to-indigo-600 rounded-xl px-6 py-4 mb-8 w-fit mx-auto">
          <h1 className="text-3xl font-bold text-white flex items-center gap-2">
            <Newspaper className="w-8 h-8 text-white" /> AI News Dashboard
          </h1>
        </div>

        <Card className="w-full shadow-lg bg-white rounded-lg">
          <YouTubeSummarizer modelEngine={modelEngine} setModelEngine={setModelEngine} />
          <Separator className="my-6" />
          <CardContent>
            <NewsSearchBar onSearch={handleSearch} allTags={allTags} activeTag={activeTag} onTagClick={handleTagClick} />
          </CardContent>
          <Separator className="mb-6" />
          <CardContent>
            <CategoryTabs activeCategory={activeCategory} onCategoryChange={handleCategoryChange} />
          </CardContent>
          <CardContent className="pt-0">
            <h3 className="text-lg font-semibold mb-4">
              News Feed{activeCategory ? ` - ${activeCategory.charAt(0).toUpperCase() + activeCategory.slice(1)}` : ""}
            </h3>
            <NewsFeed
              loading={loading}
              error={error}
              news={news}
              activeCategory={activeCategory}
              onTagClick={handleTagClick}
              onRefresh={() => fetchNews(query, activeTag, activeCategory)}
            />
          </CardContent>
        </Card>
      </div>
    </main>
  );
};

export default NewsDashboard;
