"use client";
import * as React from "react";

export function Badge({ variant = "default", className, ...props }) {
  const variants = {
    default: "bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-300",
    secondary: "bg-neutral-100 text-neutral-800 dark:bg-neutral-800 dark:text-neutral-300",
    destructive: "bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-300",
    outline: "text-neutral-900 border border-neutral-200 dark:text-neutral-100 dark:border-neutral-700",
  };

  return (
    <span
      className={`inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-semibold transition-colors focus:outline-none focus:ring-2 focus:ring-neutral-950 focus:ring-offset-2 dark:focus:ring-neutral-300 ${variants[variant]} ${className}`}
      {...props}
    />
  );
}
