"use client";

import * as React from "react";

export function Button({
  className,
  variant = "default",
  size = "default",
  type = "button",
  disabled = false,
  children,
  ...props
}) {
  const variants = {
    default:     "bg-[var(--primary)] hover:bg-[var(--primary-hover)] text-[var(--primary-fg)]",
    primary:     "bg-[var(--primary)] hover:bg-[var(--primary-hover)] text-[var(--primary-fg)]",
    secondary:   "bg-[var(--surface-muted)] hover:bg-[var(--border)] text-[var(--fg)]",
    outline:     "border border-[var(--border-strong)] hover:bg-[var(--surface-muted)] text-[var(--fg)]",
    ghost:       "hover:bg-[var(--surface-muted)] text-[var(--fg)]",
    destructive: "bg-[var(--destructive)] hover:opacity-90 text-[var(--destructive-fg)]",
  };

  const sizes = {
    default: "h-10 px-4 py-2 text-sm",
    xs:      "h-6 px-2 py-0.5 text-xs",
    sm:      "h-8 px-3 py-1 text-xs",
    lg:      "h-12 px-6 py-3 text-base",
    icon:    "h-10 w-10 p-0",
  };

  const buttonClasses = [
    "inline-flex items-center justify-center gap-2 rounded-md font-medium",
    "transition-colors duration-200 cursor-pointer",
    "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--ring)] focus-visible:ring-offset-2",
    "disabled:opacity-50 disabled:pointer-events-none",
    variants[variant] ?? variants.default,
    sizes[size] ?? sizes.default,
    className ?? "",
  ].join(" ");

  return (
    <button
      type={type}
      className={buttonClasses}
      disabled={disabled}
      {...props}
    >
      {children}
    </button>
  );
}
