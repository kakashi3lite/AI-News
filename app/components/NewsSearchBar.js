"use client";

import React from "react";
import { Filter, XCircle } from "lucide-react";
import { SearchInput } from "../../components/ui/SearchInput";
import { Button } from "../../components/ui/button";
import TagChip from "../../components/TagChip";

const NewsSearchBar = ({ onSearch, allTags, activeTag, onTagClick }) => (
  <div>
    <h3 className="text-lg font-semibold mb-4">Search & Filters</h3>
    <SearchInput
      onSearch={onSearch}
      placeholder="Search news articles by keyword..."
      label=""
      className="mb-4"
    />
    <div className="mb-2 flex items-center gap-2 text-sm text-gray-600">
      <Filter className="w-4 h-4" /> Filters:
    </div>
    <div className="flex flex-wrap gap-2 mb-6 items-center">
      {allTags.map((tag) => (
        <TagChip key={tag} label={tag} active={activeTag === tag} onClick={() => onTagClick(tag)} />
      ))}
      {activeTag && (
        <Button
          variant="secondary"
          size="sm"
          className="text-xs h-auto py-1 px-2"
          onClick={() => onTagClick("")}
        >
          <XCircle className="w-3 h-3 mr-1" /> Clear Filter
        </Button>
      )}
    </div>
  </div>
);

export default NewsSearchBar;
