"use client";
import { useCallback, useEffect, useState } from 'react';
import { Loader2, AlertCircle, Plus, Building2, Trash2 } from 'lucide-react';
import CompanyCard from './CompanyCard';
import { useUser } from '../../contexts/UserContext';
import { fetchWatchlist, fetchNews, isStaticMode } from '../../lib/clientData';

// Lightweight client-side matcher for personal watchlist items
// (works in static mode where there is no server matching).
function clientMatch(name, keywords, articles) {
  const terms = [name, ...(keywords || [])].map((t) => String(t).toLowerCase()).filter((t) => t.length >= 2);
  return articles
    .filter((a) => {
      const text = `${a.title || ''} ${a.description || ''}`.toLowerCase();
      return terms.some((t) => text.includes(t));
    })
    .slice(0, 4)
    .map((a) => ({ ...a, matchedOn: [name], matchScore: 1 }));
}

// Watchlist — tracked companies with their matched news streams + add form.
// Combines the curated/default companies with the user's personal watchlist.
export default function WatchlistView() {
  const { user, data, addToWatchlist, removeFromWatchlist, setLoginOpen } = useUser();
  const [items, setItems] = useState([]);
  const [corpus, setCorpus] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [name, setName] = useState('');
  const [keywords, setKeywords] = useState('');
  const [adding, setAdding] = useState(false);

  const load = useCallback(async () => {
    setLoading(true);
    setError('');
    try {
      const [wRes, nRes] = await Promise.all([fetchWatchlist(), fetchNews({ limit: 150 })]);
      setItems(wRes.items || []);
      setCorpus(nRes.articles || []);
      if (wRes.error) setError(wRes.error);
    } catch {
      setError('Failed to load watchlist.');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  // Merge curated items with the user's personal items.
  const personal = (data.watchlist || []).filter(
    (p) => !items.some((i) => i.name.toLowerCase() === p.name.toLowerCase())
  );
  const combined = [
    ...items,
    ...personal.map((p) => ({
      id: `local-${p.name}`,
      name: p.name,
      personal: true,
      articleCount: clientMatch(p.name, p.keywords, corpus).length,
      stories: clientMatch(p.name, p.keywords, corpus),
    })),
  ];

  const addCompany = async (e) => {
    e.preventDefault();
    if (!name.trim()) return;
    if (!user) {
      setLoginOpen(true);
      return;
    }
    setAdding(true);
    setError('');
    try {
      addToWatchlist({ name: name.trim(), keywords: keywords.split(',').map((k) => k.trim()).filter(Boolean) });
      // In server mode, also persist server-side for cross-device matching.
      if (!isStaticMode) {
        await fetch('/api/watchlist', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ name: name.trim(), keywords }),
        });
      }
      setName('');
      setKeywords('');
      await load();
    } catch {
      setError('Failed to add company.');
    } finally {
      setAdding(false);
    }
  };

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-xl font-bold text-gray-900">Watchlist</h2>
        <p className="text-sm text-gray-500 mt-0.5">
          Companies and entities you track. Stories are matched by aliases and keywords — match
          reasons are shown on each story.
        </p>
      </div>

      {/* Add company */}
      <form onSubmit={addCompany} className="rounded-xl border border-gray-200 bg-white p-4 flex flex-col sm:flex-row gap-3 items-end">
        <div className="flex-1">
          <label className="block text-xs font-medium text-gray-500 mb-1">Company / entity name</label>
          <input
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="e.g. AMD, Netflix, Palantir…"
            className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
        </div>
        <div className="flex-1">
          <label className="block text-xs font-medium text-gray-500 mb-1">
            Keywords (comma-separated)
          </label>
          <input
            value={keywords}
            onChange={(e) => setKeywords(e.target.value)}
            placeholder="e.g. ryzen, epyc, instinct…"
            className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
        </div>
        <button
          type="submit"
          disabled={adding || !name.trim()}
          className="inline-flex items-center gap-2 rounded-lg bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-700 disabled:opacity-50"
        >
          {adding ? <Loader2 className="w-4 h-4 animate-spin" /> : <Plus className="w-4 h-4" />}
          Track
        </button>
      </form>
      {!user && (
        <p className="text-xs text-gray-500">
          Sign in to save your watchlist — otherwise you can browse the default companies below.
        </p>
      )}
      {error && (
        <div className="rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700">
          {error}
        </div>
      )}

      {loading ? (
        <div className="flex justify-center items-center p-16 text-gray-500">
          <Loader2 className="h-6 w-6 animate-spin mr-2" /> Loading watchlist…
        </div>
      ) : (
        <>
          <div className="flex items-center gap-2 text-sm text-gray-500">
            <Building2 className="w-4 h-4" /> {combined.length} tracked {combined.length === 1 ? 'entity' : 'entities'}
            {personal.length > 0 && <span className="text-xs text-indigo-600">({personal.length} yours)</span>}
          </div>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {combined.map((c) => (
              <div key={c.id} className="relative">
                <CompanyCard company={c} />
                {c.personal && (
                  <button
                    onClick={() => removeFromWatchlist(c.name)}
                    className="absolute top-3 right-3 inline-flex items-center gap-1 rounded-md bg-red-50 border border-red-200 px-2 py-1 text-[11px] font-medium text-red-600 hover:bg-red-100"
                  >
                    <Trash2 className="w-3 h-3" /> Untrack
                  </button>
                )}
              </div>
            ))}
          </div>
        </>
      )}
    </div>
  );
}
