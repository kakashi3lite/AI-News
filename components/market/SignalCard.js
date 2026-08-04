import { ExternalLink, Bookmark } from 'lucide-react';
import { timeAgo } from './format';
import {
  SentimentBadge,
  ReliabilityBadge,
  SourceBadge,
  CategoryBadge,
  ImpactBadge,
  VerificationBadge,
} from './badges';
import { useUser } from '../../contexts/UserContext';

// Story card with accuracy metadata: source, reliability, sentiment, recency,
// and optional watchlist-match attribution. Bookmarking saves the story to the
// user's local research vault (powers personalized recommendations).
export default function SignalCard({ article, onTagClick, matches = [] }) {
  const { isBookmarked, toggleBookmark, logReading } = useUser();
  const saved = isBookmarked(article.id);

  const handleOpen = () => {
    logReading(article);
  };

  return (
    <article className="bg-white rounded-xl border border-gray-200 shadow-sm p-5 flex flex-col gap-3 hover:shadow-md hover:border-blue-200 transition-all">
      <div className="flex items-start justify-between gap-2">
        <div className="flex items-center gap-2 min-w-0">
          {article.image ? (
            // eslint-disable-next-line @next/next/no-img-element
            <img
              src={article.image}
              alt=""
              className="h-10 w-10 rounded-lg object-cover shrink-0 bg-gray-100"
              loading="lazy"
            />
          ) : null}
          <div className="min-w-0">
            <SourceBadge source={article.source} />
            <div className="text-[11px] text-gray-400">
              {article.publishedAt ? timeAgo(article.publishedAt) : ''}
            </div>
          </div>
        </div>
        <div className="flex items-center gap-1.5 shrink-0">
          <SentimentBadge label={article.sentimentLabel} score={article.sentimentScore} />
          <ReliabilityBadge score={article.reliabilityScore} />
          <button
            onClick={() => toggleBookmark(article)}
            className={`rounded-md p-1.5 transition-colors ${
              saved ? 'text-blue-600 bg-blue-50' : 'text-gray-300 hover:text-blue-500 hover:bg-gray-50'
            }`}
            aria-label={saved ? 'Remove bookmark' : 'Save to research'}
            title={saved ? 'Saved to your research' : 'Save to research'}
          >
            <Bookmark className={`w-4 h-4 ${saved ? 'fill-current' : ''}`} />
          </button>
        </div>
      </div>

      <a
        href={article.url}
        target="_blank"
        rel="noopener noreferrer"
        onClick={handleOpen}
        className="font-semibold text-gray-900 leading-snug hover:text-blue-600 line-clamp-2"
      >
        {article.title}
      </a>

      <p className="text-sm text-gray-600 leading-relaxed line-clamp-3">
        {article.summary || article.description}
      </p>

      {/* Impact + verification row */}
      {(article.impactLabel || article.verification) && (
        <div className="flex items-center gap-1.5 flex-wrap">
          <ImpactBadge label={article.impactLabel} score={article.impactScore} />
          <VerificationBadge level={article.verification} corroboration={article.corroboration} />
          {article.outlook && (
            <span className="text-[11px] text-gray-400 italic">{article.outlook}</span>
          )}
        </div>
      )}

      <div className="mt-auto flex items-center justify-between gap-2 pt-1">
        <div className="flex items-center gap-1.5 flex-wrap min-w-0">
          <CategoryBadge category={article.category} />
          {matches.length > 0 &&
            matches.map((m) => (
              <span
                key={m}
                className="inline-flex items-center rounded-full bg-indigo-50 border border-indigo-200 px-2 py-0.5 text-[11px] font-medium text-indigo-700"
                title="Matched watchlist company"
              >
                {m}
              </span>
            ))}
          {article.tags?.slice(0, 3).map((t) => (
            <button
              key={t}
              onClick={() => onTagClick?.(t)}
              className="text-[11px] text-blue-600 hover:underline"
            >
              #{t}
            </button>
          ))}
        </div>
        <a
          href={article.url}
          target="_blank"
          rel="noopener noreferrer"
          className="text-gray-400 hover:text-blue-600 shrink-0"
          aria-label="Open article"
        >
          <ExternalLink className="w-4 h-4" />
        </a>
      </div>
    </article>
  );
}
