// NewsCard: Modular card for displaying a news article
import React from "react";

export default function NewsCard({ article, onTagClick }) {
  return (
    <div className="bg-white dark:bg-neutral-900 rounded-lg shadow-sm border border-neutral-200 dark:border-neutral-800 p-4 flex flex-col h-full transition hover:shadow-md">
      {article.image && (
        <img
          src={article.image}
          alt={article.title}
          className="w-full h-40 object-cover rounded-md mb-3"
          loading="lazy"
        />
      )}
      <h3 className="font-semibold text-lg mb-1 text-blue-800 dark:text-blue-200 truncate" title={article.title}>
        <a href={article.url} target="_blank" rel="noopener noreferrer">{article.title}</a>
      </h3>
      <p className="text-sm text-neutral-700 dark:text-neutral-300 mb-2 line-clamp-3">{article.description}</p>
      <div className="flex flex-wrap gap-1 mb-2">
        {article.tags?.map(
          (tag) => tag && (
            <button
              key={tag}
              type="button"
              className="inline-block px-2 py-0.5 bg-blue-50 dark:bg-blue-900 text-blue-700 dark:text-blue-200 rounded-full text-xs border border-blue-200 dark:border-blue-800 hover:bg-blue-100 dark:hover:bg-blue-800 mr-1 mb-1"
              onClick={() => onTagClick?.(tag)}
              aria-label={`Filter by ${tag}`}
            >
              {tag}
            </button>
          )
        )}
      </div>
      <div className="text-xs text-neutral-400 mt-auto">
        {article.source?.name}
      </div>
    </div>
  );
}
// ---
// This card is used in the news grid. It is modular, accessible, and reusable.
// Tags are clickable for filtering. Add more metadata as needed.
