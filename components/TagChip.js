// TagChip: Modular, clickable chip for tags/categories
import React from "react";

export default function TagChip({ label, active, onClick }) {
  return (
    <button
      type="button"
      className="inline-flex items-center px-3 py-1 rounded-full text-xs font-medium border transition-colors duration-150 outline-none cursor-pointer focus-visible:ring-2"
      style={
        active
          ? {
              backgroundColor: 'var(--primary)',
              color: 'var(--primary-fg)',
              borderColor: 'var(--primary)',
            }
          : {
              backgroundColor: 'var(--surface)',
              color: 'var(--fg-muted)',
              borderColor: 'var(--border-strong)',
            }
      }
      onMouseEnter={e => {
        if (!active) {
          e.currentTarget.style.backgroundColor = 'var(--accent)';
          e.currentTarget.style.color = 'var(--accent-fg)';
        }
      }}
      onMouseLeave={e => {
        if (!active) {
          e.currentTarget.style.backgroundColor = 'var(--surface)';
          e.currentTarget.style.color = 'var(--fg-muted)';
        }
      }}
      onClick={onClick}
      aria-pressed={active}
      aria-label={`Filter by ${label}`}
    >
      {label}
    </button>
  );
}
