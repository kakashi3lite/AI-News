"use client";

import { useEffect, useMemo, useState } from 'react';
import {
  Loader2,
  AlertCircle,
  ShieldCheck,
  Zap,
  Sun,
  Moon,
  Coffee,
  Flame,
  Grid3x3,
  Sparkles,
  Building2,
  ArrowRight,
  Crown,
} from 'lucide-react';
import { useUser } from '../../contexts/UserContext';
import { fetchMeta, fetchNews, fetchWatchlist, fetchCrossword } from '../../lib/clientData';
import SignalCard from './SignalCard';
import { SentimentBadge, ImpactBadge } from './badges';

function greeting() {
  const h = new Date().getHours();
  if (h < 5) return { text: 'Burning the midnight oil', icon: Moon };
  if (h < 12) return { text: 'Good morning', icon: Sun };
  if (h < 17) return { text: 'Good afternoon', icon: Coffee };
  return { text: 'Good evening', icon: Moon };
}

function timeAgoText(iso) {
  if (!iso) return 'just now';
  const mins = Math.floor((Date.now() - new Date(iso).getTime()) / 60000);
  if (mins < 60) return `${mins}m ago`;
  const hours = Math.floor(mins / 60);
  if (hours < 24) return `${hours}h ago`;
  return `${Math.floor(hours / 24)}d ago`;
}

// Today = the morning-briefing home. Right components, right moment:
// trust strip → personal briefing → crossword habit hook → watchlist pulse → Pro.
export default function TodayView({ onOpenCrossword }) {
  const { user, data, setLoginOpen, crosswordStatus } = useUser();
  const [meta, setMeta] = useState(null);
  const [stories, setStories] = useState([]);
  const [watchlist, setWatchlist] = useState([]);
  const [crossword, setCrossword] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const xwStatus = crosswordStatus();

  useEffect(() => {
    (async () => {
      try {
        const [mRes, nRes, wRes, cRes] = await Promise.all([
          fetchMeta(),
          fetchNews({ limit: 12 }),
          fetchWatchlist(),
          fetchCrossword(),
        ]);
        setMeta(mRes.error ? null : mRes);
        setStories(nRes.articles || []);
        setWatchlist((wRes.items || []).slice(0, 8));
        setCrossword(cRes.error ? null : cRes);
        if (nRes.error) setError(nRes.error);
      } catch {
        setError('Failed to load your briefing.');
      } finally {
        setLoading(false);
      }
    })();
  }, []);

  // Personal briefing: prefer stories that match interests/watchlist for signed-in users.
  const briefing = useMemo(() => {
    if (!user) return stories.slice(0, 3);
    const interests = Object.keys(data.interests || {});
    const watchNames = (data.watchlist || []).map((w) => w.name.toLowerCase());
    const scored = stories.map((s) => {
      const text = `${s.title} ${s.description}`.toLowerCase();
      let score = s.impactScore || 0;
      for (const t of interests) if (t && text.includes(t)) score += 12;
      for (const n of watchNames) if (n && text.includes(n)) score += 15;
      return { ...s, score };
    });
    return scored.sort((a, b) => b.score - a.score).slice(0, 3);
  }, [user, data, stories]);

  const g = greeting();
  const GreetIcon = g.icon;

  if (loading) {
    return (
      <div className="flex justify-center items-center p-16 text-gray-500">
        <Loader2 className="h-6 w-6 animate-spin mr-2" /> Building your briefing…
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Greeting */}
      <div className="flex items-center gap-3">
        <GreetIcon className="w-7 h-7 text-amber-500" />
        <div>
          <h2 className="text-2xl font-bold text-gray-900">
            {g.text}
            {user ? `, ${user.name.split(' ')[0]}` : ''}
          </h2>
          <p className="text-sm text-gray-500">
            {new Date().toLocaleDateString(undefined, { weekday: 'long', month: 'long', day: 'numeric' })}
            {' · '}your market briefing
          </p>
        </div>
      </div>

      {/* Trust strip — the accuracy moat, front and center */}
      <div className="flex flex-wrap items-center gap-x-5 gap-y-1.5 text-xs text-gray-600 bg-white border border-gray-200 rounded-lg px-4 py-2.5">
        <span className="inline-flex items-center gap-1.5">
          <ShieldCheck className="w-4 h-4 text-emerald-600" />
          <b>{meta?.sources ?? 13}</b> curated sources
        </span>
        <span className="inline-flex items-center gap-1.5">
          <Zap className="w-4 h-4 text-blue-600" />
          <b>{meta?.last24h ?? '—'}</b> new stories in the last 24h
        </span>
        <span className="inline-flex items-center gap-1.5">
          <Sparkles className="w-4 h-4 text-indigo-500" />
          impact + verification scored
        </span>
        <span className="text-gray-400 ml-auto">
          data updated {meta?.generatedAt ? timeAgoText(meta.generatedAt) : 'recently'}
        </span>
      </div>

      {error && (
        <div className="rounded-xl border border-red-200 bg-red-50 p-4 flex items-center text-red-700">
          <AlertCircle className="w-5 h-5 mr-2 shrink-0" />
          <span>{error}</span>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Briefing */}
        <section className="lg:col-span-2 space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-semibold text-gray-700 uppercase tracking-wide">
              {user ? 'For you today' : "Today's top stories"}
            </h3>
            {!user && (
              <button
                onClick={() => setLoginOpen(true)}
                className="inline-flex items-center gap-1 text-xs font-medium text-blue-600 hover:underline"
              >
                Sign in for a personal briefing <ArrowRight className="w-3 h-3" />
              </button>
            )}
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {briefing.map((a, i) => (
              <div key={a.id} className={i === 0 ? 'md:col-span-2' : ''}>
                <SignalCard article={a} />
              </div>
            ))}
          </div>
        </section>

        {/* Right rail: crossword + watchlist pulse + Pro */}
        <div className="space-y-4">
          {/* Crossword habit hook */}
          <button
            onClick={onOpenCrossword}
            className="w-full text-left rounded-2xl bg-gradient-to-br from-indigo-600 to-blue-700 p-5 text-white shadow-lg hover:shadow-xl transition-all"
          >
            <div className="flex items-center justify-between mb-2">
              <Grid3x3 className="w-6 h-6" />
              {xwStatus.streak > 0 && (
                <span className="inline-flex items-center gap-1 rounded-full bg-white/20 px-2.5 py-1 text-xs font-semibold">
                  <Flame className="w-3.5 h-3.5 text-orange-300" /> {xwStatus.streak}-day streak
                </span>
              )}
            </div>
            <div className="text-lg font-bold">Today&apos;s News Crossword</div>
            <p className="text-blue-100 text-sm mt-1">
              {crossword?.wordCount ?? 10} words built from today&apos;s real headlines
            </p>
            <div className="mt-3 inline-flex items-center gap-2 rounded-lg bg-white text-indigo-700 px-4 py-2 text-sm font-semibold">
              {xwStatus.solvedToday ? 'Solved today ✓ · Play again' : 'Play now'}
              <ArrowRight className="w-4 h-4" />
            </div>
            {xwStatus.solvedToday && (
              <p className="text-blue-100 text-xs mt-2">Come back tomorrow for a fresh puzzle.</p>
            )}
          </button>

          {/* Watchlist pulse */}
          {watchlist.length > 0 && (
            <div className="rounded-xl border border-gray-200 bg-white p-4">
              <h3 className="text-sm font-semibold text-gray-700 uppercase tracking-wide mb-3 flex items-center gap-2">
                <Building2 className="w-4 h-4 text-blue-600" /> Watchlist pulse
              </h3>
              <div className="flex flex-wrap gap-2">
                {watchlist.map((w) => (
                  <span
                    key={w.id}
                    className="inline-flex items-center gap-1.5 rounded-full border border-gray-200 bg-gray-50 px-2.5 py-1 text-xs"
                  >
                    <span className="font-medium text-gray-800">{w.name}</span>
                    {w.articleCount > 0 && (
                      <span className="text-gray-400">{w.articleCount}</span>
                    )}
                  </span>
                ))}
              </div>
            </div>
          )}

          {/* Pro seam — the monetization story, tastefully visible */}
          <div className="rounded-xl border border-amber-200 bg-gradient-to-br from-amber-50 to-orange-50 p-4">
            <div className="flex items-center gap-2 mb-1">
              <Crown className="w-4 h-4 text-amber-600" />
              <span className="text-sm font-bold text-gray-900">Market Signal Pro</span>
              <span className="ml-auto rounded-full bg-amber-100 text-amber-700 text-[10px] font-semibold px-2 py-0.5">
                Coming soon
              </span>
            </div>
            <p className="text-xs text-gray-600 mt-1 leading-relaxed">
              Email digests, impact alerts, historical charts, and AI summaries. Core is — and stays —
              <b> free forever</b>.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
