"use client";
import * as React from "react";

export function Separator({ orientation = "horizontal", className, ...props }) {
  return (
    <div
      className={`shrink-0 bg-neutral-200 dark:bg-neutral-800 ${
        orientation === "horizontal" ? "h-[1px] w-full" : "h-full w-[1px]"
      } ${className}`}
      {...props}
    />
  );
}
