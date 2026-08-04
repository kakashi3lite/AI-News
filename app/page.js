'use client';

import React from "react";
import { Button } from "../components/ui/button";
import FeatureCard from "../components/FeatureCard";
import { TrendingUp, ShieldCheck, Building2, Newspaper, Mail } from "lucide-react";

export default function Page() {
  return (
    <main className="min-h-screen px-4 py-16 bg-gray-50 flex flex-col items-center">
      <h1 className="text-5xl font-bold mb-4 text-gray-900">Market Signal</h1>
      <p className="text-lg text-gray-600 mb-8 text-center max-w-2xl">
        Competitive intelligence that stays accurate: real-time market news, graded
        by source reliability and tagged with sentiment — plus company watchlists
        and a daily digest.
      </p>
      <Button variant="primary" size="lg" href="/dashboard" className="mb-12">
        Go to Dashboard
      </Button>
      <section className="grid grid-cols-1 md:grid-cols-3 gap-6 max-w-4xl w-full">
        <FeatureCard
          icon={<TrendingUp className="w-6 h-6 text-blue-500" />}
          title="Today's Signal"
          description="Themes and top stories ranked by recency, source reliability, and sentiment."
        />
        <FeatureCard
          icon={<Building2 className="w-6 h-6 text-indigo-500" />}
          title="Company Watchlist"
          description="Track competitors and partners — matched by aliases and keywords, with match reasons shown."
        />
        <FeatureCard
          icon={<ShieldCheck className="w-6 h-6 text-emerald-500" />}
          title="Source-Graded Accuracy"
          description="Every story carries its source's reliability score, sentiment label, and publish time."
        />
        <FeatureCard
          icon={<Newspaper className="w-6 h-6 text-blue-500" />}
          title="Real RSS Data"
          description="No API keys required — curated feeds keep the dashboard current out of the box."
        />
        <FeatureCard
          icon={<Mail className="w-6 h-6 text-amber-500" />}
          title="Daily Digest"
          description="A single-scroll summary of the themes, stories, and watchlist movements that matter."
        />
        <FeatureCard
          icon={<Building2 className="w-6 h-6 text-red-500" />}
          title="YouTube News Bite"
          description="Summarize earnings calls and product launches from any YouTube URL."
        />
      </section>
    </main>
  );
}
