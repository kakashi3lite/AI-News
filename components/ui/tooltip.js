// Tooltip.js: Simple accessible tooltip using shadcn-ui primitives or fallback
"use client";
import * as React from "react";

// This is a minimal tooltip implementation. Replace with shadcn-ui or Radix if available.
export function Tooltip({ children }) {
  return <span className="relative group">{children}</span>;
}

export function TooltipTrigger({ asChild, children }) {
  // asChild is ignored for simplicity
  return children;
}

export function TooltipContent({ children, side = "top" }) {
  // Show on hover using group-hover
  return (
    <span
      className="absolute z-50 px-2 py-1 rounded bg-neutral-800 text-white text-xs whitespace-nowrap opacity-0 group-hover:opacity-100 group-hover:pointer-events-auto transition-opacity duration-200 mt-1"
      style={{ left: "100%", top: "50%", transform: "translateY(-50%)" }}
      role="tooltip"
    >
      {children}
    </span>
  );
}

// ---
// This is a fallback tooltip. Replace with @/components/ui/tooltip if you add shadcn-ui or Radix UI.
