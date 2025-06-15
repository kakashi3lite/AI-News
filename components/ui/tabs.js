"use client";
import * as React from "react";

export function Tabs({ defaultValue, className, children, ...props }) {
  const [value, setValue] = React.useState(defaultValue);
  
  return (
    <div className={`relative w-full ${className}`} data-value={value} {...props}>
      {React.Children.map(children, (child) =>
        React.isValidElement(child) ? React.cloneElement(child, { value, setValue }) : child
      )}
    </div>
  );
}

export function TabsList({ className, children, value, setValue, ...props }) {
  return (
    <div 
      className={`inline-flex h-10 items-center justify-center rounded-lg bg-neutral-100 p-1 text-neutral-500 dark:bg-neutral-800 dark:text-neutral-400 ${className}`}
      role="tablist"
      {...props}
    >
      {React.Children.map(children, (child) =>
        React.isValidElement(child) ? React.cloneElement(child, { value, setValue }) : child
      )}
    </div>
  );
}

export function TabsTrigger({ value: triggerValue, children, value, setValue, className, ...props }) {
  const isActive = value === triggerValue;
  
  return (
    <button
      type="button"
      role="tab"
      aria-selected={isActive}
      className={`inline-flex items-center justify-center whitespace-nowrap rounded-md px-3 py-1.5 text-sm font-medium ring-offset-white transition-all focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-neutral-950 focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 data-[state=active]:bg-white data-[state=active]:text-neutral-950 data-[state=active]:shadow-sm dark:ring-offset-neutral-950 dark:focus-visible:ring-neutral-300 dark:data-[state=active]:bg-neutral-950 dark:data-[state=active]:text-neutral-50 ${
        isActive ? "bg-white text-neutral-950 shadow-sm dark:bg-neutral-950 dark:text-neutral-50" : ""
      } ${className}`}
      onClick={() => setValue(triggerValue)}
      {...props}
    >
      {children}
    </button>
  );
}

export function TabsContent({ value: contentValue, children, value, className, ...props }) {
  if (contentValue !== value) return null;
  
  return (
    <div
      role="tabpanel"
      className={`mt-2 ring-offset-white focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-neutral-950 focus-visible:ring-offset-2 dark:ring-offset-neutral-950 dark:focus-visible:ring-neutral-300 ${className}`}
      {...props}
    >
      {children}
    </div>
  );
}
