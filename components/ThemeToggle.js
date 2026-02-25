"use client";

import { Sun, Moon } from "lucide-react";
import { useTheme } from "../app/ThemeProvider";

export default function ThemeToggle() {
  const { theme, toggleTheme } = useTheme();

  return (
    <button
      onClick={toggleTheme}
      aria-label={theme === "dark" ? "Switch to light mode" : "Switch to dark mode"}
      className="h-9 w-9 rounded-lg flex items-center justify-center transition-colors cursor-pointer"
      style={{ color: 'var(--fg-muted)' }}
      onMouseEnter={e => { e.currentTarget.style.color = 'var(--fg)'; e.currentTarget.style.backgroundColor = 'var(--surface-muted)'; }}
      onMouseLeave={e => { e.currentTarget.style.color = 'var(--fg-muted)'; e.currentTarget.style.backgroundColor = ''; }}
    >
      {theme === "dark" ? (
        <Sun className="w-4 h-4" />
      ) : (
        <Moon className="w-4 h-4" />
      )}
    </button>
  );
}
