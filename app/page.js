'use client';

import React from "react";
import { Button } from "../components/ui/button";
import FeatureCard from "../components/FeatureCard";
import { Newspaper, Youtube, Filter } from "lucide-react";

export default function Page() {
  return (
    <main className="min-h-screen px-4 py-16 bg-gray-50 flex flex-col items-center">
      <h1 className="text-5xl font-bold mb-4">AI News Dashboard</h1>
      <p className="text-lg text-gray-600 mb-8 text-center">
        Summarize news articles and YouTube videos instantly with AI.
      </p>
      <Button variant="primary" size="lg" href="/dashboard" className="mb-12">
        Go to Dashboard
      </Button>
      <section className="grid grid-cols-1 md:grid-cols-3 gap-6 max-w-4xl w-full">
        <FeatureCard
          icon={<Newspaper className="w-6 h-6 text-blue-500" />}
          title="News Summarization"
          description="Get concise summaries of news articles with a single click."
        />
        <FeatureCard
          icon={<Youtube className="w-6 h-6 text-red-600" />}
          title="YouTube Summaries"
          description="Turn long videos into quick news bites."
        />
        <FeatureCard
          icon={<Filter className="w-6 h-6 text-green-500" />}
          title="Advanced Filters"
          description="Filter by tags, categories, and keywords."
        />
      </section>
    </main>
  );
}
