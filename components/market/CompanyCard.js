import { Building2 } from 'lucide-react';
import { timeAgo } from './format';
import { SentimentBadge, ImpactBadge } from './badges';

// Per-company card in the watchlist: latest matched stories + match attribution.
export default function CompanyCard({ company }) {
  const trendColor =
    company.sentimentTrend === 'positive'
      ? 'text-green-600'
      : company.sentimentTrend === 'negative'
        ? 'text-red-600'
        : 'text-gray-400';

  return (
    <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-5 flex flex-col">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2 min-w-0">
          <Building2 className="w-5 h-5 text-blue-600 shrink-0" />
          <h3 className="font-bold text-gray-900 truncate">{company.name}</h3>
        </div>
        <span className="text-xs text-gray-500 shrink-0">
          {company.articleCount} {company.articleCount === 1 ? 'story' : 'stories'}
        </span>
      </div>

      {(company.impactScore > 0 || company.sentimentTrend) && (
        <div className="flex items-center gap-1.5 mb-3 flex-wrap">
          {company.impactScore > 0 && <ImpactBadge label={company.impactLabel} score={company.impactScore} />}
          <span className={`text-[11px] font-medium ${trendColor}`}>
            Sentiment trend: {company.sentimentTrend || 'neutral'}
          </span>
        </div>
      )}

      {company.stories.length === 0 ? (
        <p className="text-sm text-gray-400 py-4 text-center">No recent stories matched.</p>
      ) : (
        <div className="space-y-3">
          {company.stories.slice(0, 4).map((s) => (
            <div key={s.id} className="border-t border-gray-100 pt-2.5">
              <a
                href={s.url}
                target="_blank"
                rel="noopener noreferrer"
                className="text-sm font-medium text-gray-800 hover:text-blue-600 line-clamp-2"
              >
                {s.title}
              </a>
              <div className="flex items-center gap-2 mt-1 flex-wrap">
                <span className="text-[11px] text-gray-400">
                  {s.source?.name || 'Unknown'} · {timeAgo(s.publishedAt)}
                </span>
                <SentimentBadge label={s.sentimentLabel} score={s.sentimentScore} />
              </div>
              {s.matchedOn?.length > 0 && (
                <div className="mt-1 flex items-center gap-1 flex-wrap">
                  {s.matchedOn.slice(0, 3).map((m) => (
                    <span
                      key={m}
                      className="inline-flex rounded bg-indigo-50 text-indigo-600 px-1.5 py-0.5 text-[10px] font-medium"
                    >
                      {m}
                    </span>
                  ))}
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
