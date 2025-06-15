"use client";

import * as React from "react";

// Simple Button component with variants
export function Button({
  className,
  variant = "default",
  size = "default",
  type = "button",
  disabled = false,
  children,
  ...props
}) {
  // Style mappings for variants and sizes
  const variants = {
    default: "bg-blue-600 hover:bg-blue-700 text-white",
    secondary: "bg-neutral-200 hover:bg-neutral-300 text-neutral-900 dark:bg-neutral-800 dark:hover:bg-neutral-700 dark:text-white",
    outline: "border border-neutral-300 dark:border-neutral-700 hover:bg-neutral-100 dark:hover:bg-neutral-800 text-neutral-900 dark:text-neutral-100",
    ghost: "hover:bg-neutral-100 dark:hover:bg-neutral-800 text-neutral-900 dark:text-neutral-100",
    destructive: "bg-red-600 hover:bg-red-700 text-white",
  };
  
  const sizes = {
    default: "h-10 px-4 py-2 text-sm",
    sm: "h-8 px-3 py-1 text-xs",
    lg: "h-12 px-6 py-3 text-base",
    icon: "h-10 w-10 p-0",
  };

  // Combine all the classes
  const buttonClasses = `inline-flex items-center justify-center rounded-md font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-neutral-950 focus-visible:ring-offset-2 disabled:opacity-50 disabled:pointer-events-none 
    ${variants[variant] || variants.default} 
    ${sizes[size] || sizes.default}
    ${className || ""}`;

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
