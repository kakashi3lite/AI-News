import { ShieldCheck, ArrowUpRight, ArrowDownRight, Minus } from 'lucide-react';

// Compact accuracy badges shown on every story: sentiment + source reliability.

const SENTIMENT_STYLES = {
  positive: 'bg-green-50 text-green-700 border-green-200',
  negative: 'bg-red-50 text-red-700 border-red-200',
  neutral: 'bg-gray-100 text-gray-600 border-gray-200',
};

const SENTIMENT_ICONS = {
  positive: <ArrowUpRight className="w-3 h-3" />,
  negative: <ArrowDownRight className="w-3 h-3" />,
  neutral: <Minus className="w-3 h-3" />,
};

export function SentimentBadge({ label, score }) {
  const l = label || 'neutral';
  return (
    <span
      title={`Sentiment: ${l} (score ${score ?? 0})`}
      className={`inline-flex items-center gap-0.5 rounded-full border px-2 py-0.5 text-[11px] font-medium ${SENTIMENT_STYLES[l] || SENTIMENT_STYLES.neutral}`}
    >
      {SENTIMENT_ICONS[l] || SENTIMENT_ICONS.neutral}
      {l}
    </span>
  );
}

export function ReliabilityBadge({ score }) {
  const pct = Math.round((score ?? 0.7) * 100);
  const color =
    pct >= 90
      ? 'bg-emerald-50 text-emerald-700 border-emerald-200'
      : pct >= 80
        ? 'bg-teal-50 text-teal-700 border-teal-200'
        : 'bg-amber-50 text-amber-700 border-amber-200';
  return (
    <span
      title="Source reliability score"
      className={`inline-flex items-center gap-0.5 rounded-full border px-2 py-0.5 text-[11px] font-medium ${color}`}
    >
      <ShieldCheck className="w-3 h-3" />
      {pct}
    </span>
  );
}

export function SourceBadge({ source }) {
  if (!source?.name) return null;
  return (
    <span className="inline-flex items-center text-xs font-semibold text-gray-500 truncate">
      {source.name}
    </span>
  );
}

export function CategoryBadge({ category }) {
  if (!category || category === 'general') return null;
  return (
    <span className="inline-flex items-center rounded-full bg-blue-50 border border-blue-200 px-2 py-0.5 text-[11px] font-medium text-blue-700">
      {category}
    </span>
  );
}
