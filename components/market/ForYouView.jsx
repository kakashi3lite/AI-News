'use client';

import { useEffect, useMemo, useState } from 'react';
import { Loader2, AlertCircle, Sparkles, Bookmark, Trash2, Clock, LogIn } from 'lucide-react';
import { useUser } from '../../contexts/UserContext';
import { fetchNews } from '../../lib/clientData';
import SignalCard from './SignalCard';
import { timeAgo } from './format';

// Build a lightweight interest profile from the user's research.
function buildProfile(data) {
  const tagCount = {};
  for (const b of data.bookmarks || []) {
    for (const t of b.tags || []) {
      const k = String(t).toLowerCase();
      tagCount[k] = (tagCount[k] || 0) + 1;
    }
  }
  const names = (data.watchlist || []).map((w) => w.name.toLowerCase());
  const keywords = (data.watchlist || []).flatMap((w) => w.keywords || []).map((k) => k.toLowerCase());
  return { tagCount, names, keywords };
}

function scoreStory(article, profile) {
  let score = 0;
  const reasons = [];
  const text = `${article.title || ''} ${article.description || ''}`.toLowerCase();

  for (const t of article.tags || []) {
    const c = profile.tagCount[String(t).toLowerCase()];
    if (c) {
      score += 2 + Math.min(c, 5);
      if (reasons.length < 3) reasons.push(t);
    }
  }
  for (const n of profile.names) {
    if (text.includes(n)) {
      score += 4;
      if (reasons.length < 3) reasons.push(n);
    }
  }
  for (const k of profile.keywords) {
    if (text.includes(k)) score += 1.5;
  }
  return { score, reasons };
}

export default function ForYouView() {
  const { user, data, setLoginOpen, isBookmarked, toggleBookmark, removeFromWatchlist } = useUser();
  const [articles, setArticles] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    (async () => {
      try {
        const res = await fetchNews({ limit: 120 });
        setArticles(res.articles || []);
        if (res.error) setError(res.error);
      } catch {
        setError('Failed to load stories.');
      } finally {
        setLoading(false);
      }
    })();
  }, []);

  const profile = useMemo(() => buildProfile(data), [data]);

  const recommended = useMemo(() => {
    const bookmarked = new Set((data.bookmarks || []).map((b) => b.id));
    const scored = articles
      .filter((a) => !bookmarked.has(a.id))
      .map((a) => ({ ...a, ...scoreStory(a, profile) }));
    scored.sort((a, b) => b.score - a.score || new Date(b.publishedAt) - new Date(a.publishedAt));
    return scored.slice(0, 12);
  }, [articles, profile, data.bookmarks]);

  const hasInterests =
    (data.bookmarks?.length || 0) + (data.watchlist?.length || 0) + (data.history?.length || 0) > 0;

  if (loading) {
    return (
      <div className="flex justify-center items-center p-16 text-gray-500">
        <Loader2 className="h-6 w-6 animate-spin mr-2" /> Building your recommendations…
      </div>
    );
  }

  if (error) {
    return (
      <div className="rounded-xl border border-red-200 bg-red-50 p-4 flex items-center text-red-700">
        <AlertCircle className="w-5 h-5 mr-2 shrink-0" />
        <span>{error}</span>
      </div>
    );
  }

  // Not signed in → explain the value and show what's popular.
  if (!user) {
    return (
      <div className="space-y-6 max-w-3xl">
        <div className="rounded-2xl border border-blue-200 bg-blue-50 p-8 text-center">
          <Sparkles className="w-10 h-10 text-blue-600 mx-auto mb-3" />
          <h2 className="text-xl font-bold text-gray-900">Your personal market feed</h2>
          <p className="text-sm text-gray-600 mt-2 max-w-md mx-auto">
            Sign in to save stories, build your own watchlist, and get a “For You” feed
            that learns what you research — every time you log in.
          </p>
          <button
            onClick={() => setLoginOpen(true)}
            className="mt-5 inline-flex items-center gap-2 rounded-lg bg-blue-600 px-5 py-2.5 text-sm font-semibold text-white hover:bg-blue-700"
          >
            <LogIn className="w-4 h-4" /> Sign in / Create profile
          </button>
        </div>

        <div>
          <h3 className="text-sm font-semibold text-gray-700 uppercase tracking-wide mb-3">
            Popular right now
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {articles.slice(0, 8).map((a) => (
              <SignalCard key={a.id} article={a} />
            ))}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-8">
      <div>
        <h2 className="text-xl font-bold text-gray-900 flex items-center gap-2">
          <Sparkles className="w-5 h-5 text-blue-600" /> For You, {user.name.split(' ')[0]}
        </h2>
        <p className="text-sm text-gray-500 mt-0.5">
          {hasInterests
            ? 'Ranked from your saved stories, watchlist, and reading history.'
            : 'Save a few stories or add watchlist companies and this feed will learn your interests.'}
        </p>
      </div>

      {recommended.length === 0 ? (
        <div className="rounded-xl border border-dashed border-gray-300 bg-white p-10 text-center text-gray-500">
          Nothing to recommend yet — bookmark stories or add watchlist companies to tune your feed.
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {recommended.map((a) => (
            <div key={a.id} className="relative">
              <SignalCard article={a} />
              {a.reasons.length > 0 && (
                <div className="mt-2 flex items-center gap-1 flex-wrap text-[11px] text-gray-500">
                  <span className="font-medium">Because you follow:</span>
                  {a.reasons.slice(0, 3).map((r) => (
                    <span key={r} className="rounded bg-blue-50 text-blue-700 px-1.5 py-0.5 font-medium">
                      {r}
                    </span>
                  ))}
                </div>
              )}
            </div>
          ))}
        </div>
      )}

      {/* Saved research */}
      <section className="rounded-xl border border-gray-200 bg-white p-5">
        <h3 className="text-sm font-semibold text-gray-700 uppercase tracking-wide mb-3 flex items-center gap-2">
          <Bookmark className="w-4 h-4 text-blue-600" /> Your saved research
          <span className="text-gray-400 normal-case text-xs">({data.bookmarks?.length || 0})</span>
        </h3>
        {data.bookmarks?.length === 0 ? (
          <p className="text-sm text-gray-400 py-3">
            Bookmark stories with the bookmark icon on any card to build your research library.
          </p>
        ) : (
          <ul className="divide-y divide-gray-100">
            {data.bookmarks.map((b) => (
              <li key={b.id} className="py-2.5 flex items-start justify-between gap-3">
                <a
                  href={b.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-sm font-medium text-gray-800 hover:text-blue-600 line-clamp-2"
                >
                  {b.title}
                </a>
                <div className="flex items-center gap-2 shrink-0">
                  <span className="text-[11px] text-gray-400">{b.source}</span>
                  <button
                    onClick={() => toggleBookmark(b)}
                    className="text-gray-400 hover:text-red-500"
                    aria-label="Remove bookmark"
                  >
                    <Trash2 className="w-4 h-4" />
                  </button>
                </div>
              </li>
            ))}
          </ul>
        )}
      </section>

      {/* Recently read */}
      {data.history?.length > 0 && (
        <section>
          <h3 className="text-sm font-semibold text-gray-700 uppercase tracking-wide mb-3 flex items-center gap-2">
            <Clock className="w-4 h-4 text-gray-500" /> Recently read
          </h3>
          <div className="space-y-1.5">
            {data.history.slice(0, 6).map((h) => (
              <div key={h.id} className="flex items-center justify-between text-sm text-gray-600">
                <span className="truncate mr-3">{h.title}</span>
                <span className="text-[11px] text-gray-400 shrink-0">{timeAgo(h.at)}</span>
              </div>
            ))}
          </div>
        </section>
      )}

      {/* Personal watchlist management */}
      {data.watchlist?.length > 0 && (
        <section>
          <h3 className="text-sm font-semibold text-gray-700 uppercase tracking-wide mb-3">
            Your watchlist
          </h3>
          <div className="flex flex-wrap gap-2">
            {data.watchlist.map((w) => (
              <span
                key={w.name}
                className="inline-flex items-center gap-2 rounded-full border border-gray-200 bg-gray-50 px-3 py-1 text-sm"
              >
                <span className="font-medium text-gray-900">{w.name}</span>
                <button
                  onClick={() => removeFromWatchlist(w.name)}
                  className="text-gray-400 hover:text-red-500"
                  aria-label={`Remove ${w.name}`}
                >
                  <Trash2 className="w-3.5 h-3.5" />
                </button>
              </span>
            ))}
          </div>
        </section>
      )}
    </div>
  );
}
