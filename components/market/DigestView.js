"use client";
import { useEffect, useState } from 'react';
import { Loader2, AlertCircle, Mail, TrendingUp, Building2 } from 'lucide-react';
import SignalCard from './SignalCard';
import { SentimentBadge } from './badges';
import { formatDate } from './format';
import { fetchDigest } from '../../lib/clientData';

// Daily digest — a single-scroll "what matters today" summary.
export default function DigestView() {
  const [digest, setDigest] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    (async () => {
      try {
        const data = await fetchDigest();
        if (data.error) setError(data.error);
        else setDigest(data);
      } catch {
        setError('Failed to load digest.');
      } finally {
        setLoading(false);
      }
    })();
  }, []);

  if (loading) {
    return (
      <div className="flex justify-center items-center p-16 text-gray-500">
        <Loader2 className="h-6 w-6 animate-spin mr-2" /> Building today&apos;s digest…
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

  return (
    <div className="space-y-6">
      <div className="flex items-start gap-3">
        <Mail className="w-6 h-6 text-blue-600 mt-1" />
        <div>
          <h2 className="text-xl font-bold text-gray-900">Daily Digest</h2>
          <p className="text-sm text-gray-500 mt-0.5">
            {digest?.generatedAt ? formatDate(digest.generatedAt) : ''} — the themes, stories, and
            watchlist movements that matter right now.
          </p>
        </div>
      </div>

      {/* Themes pulse */}
      {digest?.themes?.length > 0 && (
        <section className="rounded-xl border border-gray-200 bg-white p-5">
          <h3 className="text-sm font-semibold text-gray-700 uppercase tracking-wide mb-3 flex items-center gap-2">
            <TrendingUp className="w-4 h-4 text-blue-600" /> Theme Pulse
          </h3>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
            {digest.themes.slice(0, 8).map((t) => (
              <div key={t.slug} className="rounded-lg bg-gray-50 border border-gray-100 p-3">
                <div className="font-semibold text-gray-900 capitalize truncate">{t.name}</div>
                <div className="flex items-center gap-2 mt-1 text-xs text-gray-500">
                  <span>{t.articleCount} stories</span>
                  {t.velocity > 0 && <span className="text-green-600 font-medium">+{t.velocity}</span>}
                  <SentimentBadge label={t.sentimentLabel} score={t.sentimentScore} />
                </div>
              </div>
            ))}
          </div>
        </section>
      )}

      {/* Watchlist pulse */}
      {digest?.watchlist?.length > 0 && (
        <section className="rounded-xl border border-gray-200 bg-white p-5">
          <h3 className="text-sm font-semibold text-gray-700 uppercase tracking-wide mb-3 flex items-center gap-2">
            <Building2 className="w-4 h-4 text-blue-600" /> Watchlist Pulse
          </h3>
          <div className="flex flex-wrap gap-2">
            {digest.watchlist.map((w) => (
              <span
                key={w.id}
                className="inline-flex items-center gap-2 rounded-full border border-gray-200 bg-gray-50 px-3 py-1 text-sm"
              >
                <span className="font-medium text-gray-900">{w.name}</span>
                <span className="text-gray-500">{w.articleCount} stories</span>
              </span>
            ))}
          </div>
        </section>
      )}

      {/* Top stories */}
      <section>
        <h3 className="text-sm font-semibold text-gray-700 uppercase tracking-wide mb-3">
          Today&apos;s Top Stories
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {digest?.stories?.map((a) => (
            <SignalCard key={a.id} article={a} />
          ))}
        </div>
      </section>
    </div>
  );
}
