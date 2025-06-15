"use client";
import * as React from "react";

export function DropdownMenu({ children }) {
  return <div className="relative inline-block text-left">{children}</div>;
}

export function DropdownMenuTrigger({ children, asChild, ...props }) {
  // The asChild prop is ignored for simplicity
  return (
    <button
      type="button"
      className="inline-flex w-full justify-center gap-x-1.5 rounded-md bg-white px-3 py-2 text-sm font-semibold text-gray-900 shadow-sm ring-1 ring-inset ring-gray-300 hover:bg-gray-50"
      {...props}
    >
      {children}
    </button>
  );
}

export function DropdownMenuContent({ children, align = "end", ...props }) {
  return (
    <div
      className={`absolute ${
        align === "end" ? "right-0" : "left-0"
      } z-10 mt-2 w-56 origin-top-right rounded-md bg-white shadow-lg ring-1 ring-black ring-opacity-5 focus:outline-none`}
      {...props}
    >
      <div className="py-1">{children}</div>
    </div>
  );
}

export function DropdownMenuItem({ children, ...props }) {
  return (
    <button
      type="button"
      className="block w-full px-4 py-2 text-left text-sm text-gray-700 hover:bg-gray-100 hover:text-gray-900"
      {...props}
    >
      {children}
    </button>
  );
}
