"use client";
import { useEffect, useState } from 'react';
import { Loader2, AlertCircle, RefreshCw, Sparkles, X } from 'lucide-react';
import SignalCard from './SignalCard';
import ThemeCluster from './ThemeCluster';
import { capitalize } from './format';
import { fetchNews, fetchThemes, triggerIngest } from '../../lib/clientData';

// "Today's Signal" — the default landing view: trending themes + top stories
// ranked by recency, source reliability, and sentiment magnitude.
export default function SignalView() {
  const [themes, setThemes] = useState([]);
  const [stories, setStories] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [selectedTheme, setSelectedTheme] = useState(null);
  const [refreshing, setRefreshing] = useState(false);

  const load = async () => {
    setLoading(true);
    setError('');
    try {
      const [tRes, sRes] = await Promise.all([fetchThemes(), fetchNews({ limit: 12 })]);
      setThemes(tRes.themes || []);
      setStories(sRes.articles || []);
      if (tRes.error) setError(tRes.error);
      if (sRes.error) setError(sRes.error);
    } catch {
      setError('Failed to load the signal. Is the dev server running?');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handleRefresh = async () => {
    setRefreshing(true);
    try {
      await triggerIngest();
      await load();
    } catch {
      setError('Refresh failed.');
    } finally {
      setRefreshing(false);
    }
  };

  const filtered = selectedTheme
    ? stories.filter((s) =>
        `${s.title} ${s.description}`.toLowerCase().includes(selectedTheme.name.toLowerCase())
      )
    : stories;

  return (
    <div className="space-y-6">
      <div className="flex items-start justify-between gap-4">
        <div>
          <h2 className="text-xl font-bold text-gray-900 flex items-center gap-2">
            <Sparkles className="w-5 h-5 text-blue-600" /> Today&apos;s Signal
          </h2>
          <p className="text-sm text-gray-500 mt-0.5">
            What&apos;s moving in your tracked market — ranked by recency, source reliability, and
            sentiment.
          </p>
        </div>
        <button
          onClick={handleRefresh}
          disabled={refreshing}
          className="inline-flex items-center gap-2 rounded-lg border border-gray-300 bg-white px-3 py-2 text-sm font-medium text-gray-700 hover:bg-gray-50 disabled:opacity-60 shrink-0"
        >
          {refreshing ? <Loader2 className="w-4 h-4 animate-spin" /> : <RefreshCw className="w-4 h-4" />}
          Refresh data
        </button>
      </div>

      {loading ? (
        <div className="flex justify-center items-center p-16 text-gray-500">
          <Loader2 className="h-6 w-6 animate-spin mr-2" /> Loading real-time market signal…
        </div>
      ) : error ? (
        <div className="rounded-xl border border-red-200 bg-red-50 p-4 flex items-center text-red-700">
          <AlertCircle className="w-5 h-5 mr-2 shrink-0" />
          <span>{error}</span>
        </div>
      ) : (
        <>
          {themes.length > 0 && (
            <section>
              <h3 className="text-sm font-semibold text-gray-700 uppercase tracking-wide mb-3">
                Trending Themes
              </h3>
              <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-3">
                {themes.map((t) => (
                  <ThemeCluster
                    key={t.slug}
                    theme={t}
                    active={selectedTheme?.slug === t.slug}
                    onSelect={() =>
                      setSelectedTheme(selectedTheme?.slug === t.slug ? null : t)
                    }
                  />
                ))}
              </div>
            </section>
          )}

          <section>
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-sm font-semibold text-gray-700 uppercase tracking-wide">
                Top Stories
                {selectedTheme ? ` — ${capitalize(selectedTheme.name)}` : ''}
              </h3>
              {selectedTheme && (
                <button
                  onClick={() => setSelectedTheme(null)}
                  className="inline-flex items-center gap-1 text-xs text-blue-600 hover:underline"
                >
                  <X className="w-3 h-3" /> Clear filter
                </button>
              )}
            </div>
            {filtered.length === 0 ? (
              <div className="rounded-xl border border-dashed border-gray-300 bg-white p-10 text-center text-gray-500">
                No stories match this theme yet.
              </div>
            ) : (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {filtered.map((a) => (
                  <SignalCard key={a.id} article={a} />
                ))}
              </div>
            )}
          </section>
        </>
      )}
    </div>
  );
}
