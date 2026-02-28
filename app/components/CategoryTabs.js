"use client";

import React from "react";
import { Tabs, TabsList, TabsTrigger } from "../../components/ui/tabs";

const CategoryTabs = ({ activeCategory, onCategoryChange }) => (
  <Tabs value={activeCategory} onValueChange={onCategoryChange} className="w-full">
    <TabsList className="grid w-full grid-cols-2 sm:grid-cols-3 md:grid-cols-5 gap-2">
      <TabsTrigger value="general">General</TabsTrigger>
      <TabsTrigger value="technology">Technology</TabsTrigger>
      <TabsTrigger value="business">Business</TabsTrigger>
      <TabsTrigger value="sports">Sports</TabsTrigger>
      <TabsTrigger value="world">World</TabsTrigger>
    </TabsList>
  </Tabs>
);

export default CategoryTabs;
