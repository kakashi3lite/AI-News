// NewsCard: Modular card for displaying a news article
import React from "react";

export default function NewsCard({ article, onTagClick }) {
  return (
    <div
      className="rounded-xl border p-4 flex flex-col h-full transition-all duration-200 hover:-translate-y-0.5"
      style={{
        backgroundColor: 'var(--surface)',
        borderColor: 'var(--border)',
        boxShadow: 'var(--shadow-card)',
      }}
      onMouseEnter={e => { e.currentTarget.style.boxShadow = 'var(--shadow-card-hover)'; }}
      onMouseLeave={e => { e.currentTarget.style.boxShadow = 'var(--shadow-card)'; }}
    >
      {article.image && (
        <img
          src={article.image}
          alt={article.title}
          className="w-full h-44 object-cover rounded-lg mb-3"
          loading="lazy"
        />
      )}
      <h3
        className="font-semibold text-base mb-1.5 truncate"
        title={article.title}
        style={{ color: 'var(--primary)' }}
      >
        <a
          href={article.url}
          target="_blank"
          rel="noopener noreferrer"
          className="hover:underline underline-offset-2"
        >
          {article.title}
        </a>
      </h3>
      <p
        className="text-sm mb-3 line-clamp-3 leading-relaxed"
        style={{ color: 'var(--fg-muted)' }}
      >
        {article.description}
      </p>
      <div className="flex flex-wrap gap-1 mb-3">
        {article.tags?.map(
          (tag) => tag && (
            <button
              key={tag}
              type="button"
              className="inline-block px-2 py-0.5 rounded-full text-xs border transition-colors duration-150 cursor-pointer focus-visible:outline-none focus-visible:ring-2"
              style={{
                backgroundColor: 'var(--accent)',
                color: 'var(--accent-fg)',
                borderColor: 'transparent',
              }}
              onMouseEnter={e => {
                e.currentTarget.style.backgroundColor = 'var(--primary)';
                e.currentTarget.style.color = 'var(--primary-fg)';
              }}
              onMouseLeave={e => {
                e.currentTarget.style.backgroundColor = 'var(--accent)';
                e.currentTarget.style.color = 'var(--accent-fg)';
              }}
              onClick={() => onTagClick?.(tag)}
              aria-label={`Filter by ${tag}`}
            >
              {tag}
            </button>
          )
        )}
      </div>
      <div className="text-xs mt-auto font-medium" style={{ color: 'var(--fg-subtle)' }}>
        {article.source?.name}
      </div>
    </div>
  );
}
