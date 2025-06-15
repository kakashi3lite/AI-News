"use client";

import React, { useState } from "react";
import { Input } from "./input";
import { Label } from "./label";
import { Search, ArrowRight } from "lucide-react";

// Dedicated search bar component for global news queries
export function SearchInput({ onSearch, placeholder = "Search...", label = "", className = "" }) {
  const [query, setQuery] = useState("");

  const handleSubmit = (e) => {
    e.preventDefault();
    if (onSearch) onSearch(query);
  };

  return (
    <div className={`space-y-2 min-w-[300px] ${className}`}> 
      {label && <Label htmlFor="search-input">{label}</Label>}
      <form onSubmit={handleSubmit} className="relative">
        <Input
          id="search-input"
          className="peer pl-9 pr-9"
          placeholder={placeholder}
          type="search"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
        <div className="pointer-events-none absolute inset-y-0 left-0 flex items-center pl-3 text-neutral-500 dark:text-neutral-400">
          <Search size={16} strokeWidth={2} />
        </div>
        <button
          type="submit"
          aria-label="Submit search"
          className="absolute inset-y-0 right-0 flex h-full w-9 items-center justify-center rounded-e-lg text-neutral-500 dark:text-neutral-400 hover:text-neutral-700 focus:outline-none focus:ring-2 focus:ring-offset-2 disabled:opacity-50"
        >
          <ArrowRight size={16} strokeWidth={2} aria-hidden="true" />
        </button>
      </form>
    </div>
  );
}
