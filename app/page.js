'use client';

import React from "react";
import { Button } from "../components/ui/button";
import FeatureCard from "../components/FeatureCard";
import { Newspaper, Youtube, Filter, ArrowRight } from "lucide-react";

export default function Page() {
  return (
    <main className="flex flex-col">
      {/* Hero Section */}
      <section className="relative isolate overflow-hidden flex flex-col items-center justify-center px-4 py-24 sm:py-32 text-center">
        {/* Radial gradient background */}
        <div
          className="absolute inset-0 -z-10"
          style={{
            background: `
              radial-gradient(ellipse 80% 60% at 50% -10%,
                color-mix(in srgb, var(--primary) 15%, transparent),
                transparent 70%),
              var(--bg-subtle)
            `,
          }}
        />
        {/* Subtle grid overlay */}
        <div
          className="absolute inset-0 -z-10 opacity-[0.03]"
          style={{
            backgroundImage: `
              linear-gradient(var(--fg) 1px, transparent 1px),
              linear-gradient(90deg, var(--fg) 1px, transparent 1px)
            `,
            backgroundSize: '48px 48px',
          }}
        />

        {/* Badge pill */}
        <div
          className="inline-flex items-center gap-2 px-3 py-1 rounded-full border text-xs font-medium mb-6"
          style={{ color: 'var(--fg-muted)', backgroundColor: 'var(--surface)', borderColor: 'var(--border)' }}
        >
          <span className="w-1.5 h-1.5 rounded-full animate-pulse" style={{ backgroundColor: 'var(--primary)' }} />
          Powered by Claude AI
        </div>

        <h1
          className="font-display text-4xl sm:text-5xl lg:text-6xl font-extrabold tracking-tight mb-4 max-w-3xl"
          style={{ color: 'var(--fg)' }}
        >
          Your AI-Powered{' '}
          <span style={{ color: 'var(--primary)' }}>News Intelligence</span>{' '}
          Hub
        </h1>

        <p
          className="text-base sm:text-lg mb-10 max-w-xl leading-relaxed"
          style={{ color: 'var(--fg-muted)' }}
        >
          Summarize news articles and YouTube videos instantly. Filter by topic,
          search by keyword, and stay informed in seconds.
        </p>

        <div className="flex flex-col sm:flex-row items-center gap-3">
          <Button variant="default" size="lg" onClick={() => window.location.href = '/dashboard'}>
            Go to Dashboard <ArrowRight className="ml-2 w-4 h-4" />
          </Button>
          <Button variant="ghost" size="lg" onClick={() => window.location.href = '/dashboard'}>
            See how it works
          </Button>
        </div>
      </section>

      {/* Feature Cards Section */}
      <section className="px-4 pb-24 max-w-5xl mx-auto w-full">
        <p
          className="text-center text-xs font-semibold uppercase tracking-widest mb-8"
          style={{ color: 'var(--fg-subtle)' }}
        >
          What you can do
        </p>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
          <FeatureCard
            icon={<Newspaper className="w-5 h-5" />}
            title="News Summarization"
            description="Get concise AI summaries of any news article with a single click."
          />
          <FeatureCard
            icon={<Youtube className="w-5 h-5" />}
            title="YouTube Summaries"
            description="Turn long videos into quick news bites — paste a URL and done."
          />
          <FeatureCard
            icon={<Filter className="w-5 h-5" />}
            title="Advanced Filters"
            description="Filter by tags, categories, and keywords to find what matters."
          />
        </div>
      </section>
    </main>
  );
}
