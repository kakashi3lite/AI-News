import { TrendingUp } from 'lucide-react';
import { SentimentBadge, ImpactBadge } from './badges';

// A clickable theme cluster pill with story count, velocity, sentiment, and impact.
export default function ThemeCluster({ theme, active, onSelect }) {
  return (
    <button
      onClick={() => onSelect?.(theme)}
      className={`text-left rounded-xl border px-4 py-3 transition-all ${
        active
          ? 'bg-blue-50 border-blue-300 shadow-sm'
          : 'bg-white border-gray-200 hover:border-blue-200 hover:shadow-sm'
      }`}
    >
      <div className="flex items-center gap-2">
        <TrendingUp className={`w-4 h-4 ${active ? 'text-blue-600' : 'text-gray-400'}`} />
        <span className="font-semibold text-gray-900 capitalize truncate">{theme.name}</span>
      </div>
      <div className="flex items-center gap-2 mt-1.5 flex-wrap">
        <span className="text-xs text-gray-500">{theme.articleCount} stories</span>
        {theme.velocity > 0 && (
          <span className="text-xs font-medium text-green-600">+{theme.velocity}</span>
        )}
        <SentimentBadge label={theme.sentimentLabel} score={theme.sentimentScore} />
        {theme.impactLabel && <ImpactBadge label={theme.impactLabel} score={theme.impactScore} />}
      </div>
    </button>
  );
}
