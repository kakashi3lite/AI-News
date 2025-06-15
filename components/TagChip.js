// TagChip: Modular, clickable chip for tags/categories
import React from "react";

export default function TagChip({ label, active, onClick }) {
  return (
    <button
      type="button"
      className={`inline-flex items-center px-3 py-1 rounded-full text-xs font-medium border mr-2 mb-2 transition-colors
        ${active ? "bg-blue-600 text-white border-blue-600" : "bg-white dark:bg-neutral-900 border-neutral-300 dark:border-neutral-700 text-blue-700 dark:text-blue-200 hover:bg-blue-50 dark:hover:bg-blue-800"}`}
      style={{ outline: "none" }}
      onClick={onClick}
      aria-pressed={active}
      aria-label={`Filter by ${label}`}
    >
      {label}
    </button>
  );
}
// ---
// This chip is used for tag/category filtering in the news dashboard.
// It is accessible, modular, and can be reused for any filter chips.
