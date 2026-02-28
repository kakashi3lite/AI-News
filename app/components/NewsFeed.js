"use client";

import React from "react";
import { Loader2, AlertCircle } from "lucide-react";
import { Button } from "../../components/ui/button";
import { Card, CardContent, CardFooter } from "../../components/ui/card";
import NewsCard from "../../components/NewsCard";

const NewsFeed = ({ loading, error, news, activeCategory, onTagClick, onRefresh }) => {
  if (loading) {
    return (
      <div className="flex justify-center items-center p-10">
        <Loader2 className="h-8 w-8 animate-spin text-blue-500" />
        <span className="ml-3 text-gray-500">Loading news...</span>
      </div>
    );
  }

  if (error) {
    return (
      <Card className="mt-4 p-4 bg-red-50 border-red-200">
        <div className="flex items-center text-red-700">
          <AlertCircle className="w-5 h-5 mr-2" />
          <span>{error}</span>
        </div>
      </Card>
    );
  }

  if (news.length === 0) {
    return (
      <Card className="mt-6 bg-gray-50 border-dashed border-gray-300">
        <CardContent className="text-center text-gray-500 p-6">
          No news articles found. Try a different search or clear filters.
        </CardContent>
        <CardFooter className="flex justify-center pb-6">
          <Button variant="outline" onClick={onRefresh}>Refresh</Button>
        </CardFooter>
      </Card>
    );
  }

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
      {news.map((article) => (
        <NewsCard key={article.id} article={article} onTagClick={onTagClick} />
      ))}
    </div>
  );
};

export default NewsFeed;
