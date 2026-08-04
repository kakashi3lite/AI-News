"use client";
import { useCallback, useEffect, useState } from 'react';
import { Search, Loader2, AlertCircle, XCircle, Filter } from 'lucide-react';
import SignalCard from './SignalCard';
import { capitalize } from './format';
import { fetchNews } from '../../lib/clientData';

const CATEGORIES = ['general', 'technology', 'business', 'politics', 'science', 'world'];

// "All Stories" — searchable, filterable feed over the ingested market data.
export default function StoriesView() {
  const [query, setQuery] = useState('');
  const [appliedQuery, setAppliedQuery] = useState('');
  const [category, setCategory] = useState('');
  const [tag, setTag] = useState('');
  const [stories, setStories] = useState([]);
  const [allTags, setAllTags] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  const fetchStories = useCallback(async () => {
    setLoading(true);
    setError('');
    try {
      const data = await fetchNews({
        query: appliedQuery,
        category,
        tag,
        limit: 30,
      });
      if (data.error) {
        setError(data.error);
        setStories([]);
      } else {
        setStories(data.articles || []);
        const tagsSet = new Set();
        for (const a of data.articles || []) {
          for (const t of a.tags || []) tagsSet.add(t);
        }
        setAllTags(Array.from(tagsSet).sort().slice(0, 12));
      }
    } catch {
      setError('Failed to fetch stories.');
      setStories([]);
    } finally {
      setLoading(false);
    }
  }, [appliedQuery, category, tag]);

  useEffect(() => {
    fetchStories();
  }, [fetchStories]);

  const handleSearch = (e) => {
    e.preventDefault();
    setAppliedQuery(query);
  };

  const handleTagClick = (t) => setTag((prev) => (prev === t ? '' : t));
  const clearAll = () => {
    setQuery('');
    setAppliedQuery('');
    setCategory('');
    setTag('');
  };

  const hasFilters = Boolean(appliedQuery || category || tag);

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-xl font-bold text-gray-900">All Stories</h2>
        <p className="text-sm text-gray-500 mt-0.5">
          Search and filter the ingested market feed. Every story carries source, reliability, and
          sentiment metadata.
        </p>
      </div>

      {/* Search */}
      <form onSubmit={handleSearch} className="flex gap-2">
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search companies, topics, keywords…"
            className="w-full rounded-lg border border-gray-300 bg-white pl-9 pr-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          />
        </div>
        <button
          type="submit"
          className="rounded-lg bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-700"
        >
          Search
        </button>
        {hasFilters && (
          <button
            type="button"
            onClick={clearAll}
            className="inline-flex items-center gap-1 rounded-lg border border-gray-300 px-3 py-2 text-sm text-gray-600 hover:bg-gray-50"
          >
            <XCircle className="w-4 h-4" /> Clear
          </button>
        )}
      </form>

      {/* Category tabs */}
      <div className="flex flex-wrap gap-2">
        {CATEGORIES.map((c) => (
          <button
            key={c}
            onClick={() => setCategory(category === c ? '' : c)}
            className={`rounded-full border px-3 py-1 text-sm font-medium transition-colors ${
              category === c
                ? 'bg-blue-600 border-blue-600 text-white'
                : 'bg-white border-gray-300 text-gray-600 hover:border-blue-300'
            }`}
          >
            {capitalize(c)}
          </button>
        ))}
      </div>

      {/* Active filters */}
      {hasFilters && (
        <div className="flex items-center gap-2 text-sm text-gray-600 bg-blue-50 border border-blue-200 rounded-lg px-3 py-2">
          <Filter className="w-4 h-4 text-blue-600" />
          {appliedQuery && <span>Search: <b>{appliedQuery}</b></span>}
          {category && <span>Category: <b>{capitalize(category)}</b></span>}
          {tag && <span>Tag: <b>{tag}</b></span>}
        </div>
      )}

      {/* Tag chips */}
      {allTags.length > 0 && (
        <div className="flex flex-wrap gap-2">
          {allTags.map((t) => (
            <button
              key={t}
              onClick={() => handleTagClick(t)}
              className={`rounded-full border px-2.5 py-0.5 text-xs font-medium ${
                tag === t
                  ? 'bg-indigo-600 border-indigo-600 text-white'
                  : 'bg-white border-gray-300 text-indigo-600 hover:border-indigo-400'
              }`}
            >
              #{t}
            </button>
          ))}
        </div>
      )}

      {/* Results */}
      {loading ? (
        <div className="flex justify-center items-center p-16 text-gray-500">
          <Loader2 className="h-6 w-6 animate-spin mr-2" /> Loading stories…
        </div>
      ) : error ? (
        <div className="rounded-xl border border-red-200 bg-red-50 p-4 flex items-center text-red-700">
          <AlertCircle className="w-5 h-5 mr-2 shrink-0" />
          <span>{error}</span>
        </div>
      ) : stories.length === 0 ? (
        <div className="rounded-xl border border-dashed border-gray-300 bg-white p-10 text-center text-gray-500">
          No stories found. Try a different search or clear filters.
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {stories.map((a) => (
            <SignalCard key={a.id} article={a} onTagClick={handleTagClick} />
          ))}
        </div>
      )}
    </div>
  );
}
