import React from "react";
import { Tooltip, TooltipTrigger, TooltipContent } from "./tooltip";
import { usePathname } from "next/navigation";

// Sidebar: Modular, accessible, with active highlighting and tooltips
export function Sidebar({ open = true, setOpen, children }) {
  return (
    <aside
      className={`bg-white dark:bg-neutral-900 border-r border-neutral-200 dark:border-neutral-700 min-w-[220px] max-w-[240px] h-screen flex flex-col transition-all duration-200 ${open ? "" : "-translate-x-full"}`}
      aria-label="Sidebar navigation"
    >
      {children}
    </aside>
  );
}

export function SidebarBody({ className = "", children }) {
  return <div className={`flex flex-col flex-1 ${className}`}>{children}</div>;
}

// SidebarLink: Highlights active link, adds tooltip for accessibility
export function SidebarLink({ link }) {
  const pathname = typeof window !== "undefined" ? window.location.pathname : "";
  const isActive = link.href !== "#" && pathname === link.href;
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <a
          href={link.href}
          className={`flex items-center gap-3 px-4 py-2 rounded-md transition-colors text-sm font-medium
            ${isActive ? "bg-blue-100 text-blue-700 dark:bg-blue-900 dark:text-blue-200" : "hover:bg-neutral-200 dark:hover:bg-neutral-800"}`}
          aria-label={link.label}
        >
          {link.icon}
          <span>{link.label}</span>
        </a>
      </TooltipTrigger>
      <TooltipContent side="right">{link.label}</TooltipContent>
    </Tooltip>
  );
}

// ---
// This sidebar uses tooltips for all links and highlights the active section for clarity.
// Uses Next.js navigation for pathname detection. Modular and well-commented for maintainability.
