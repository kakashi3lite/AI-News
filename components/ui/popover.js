"use client";

import * as React from "react";

// Context to share open state and toggle function
const PopoverContext = React.createContext({
  open: false,
  onOpenChange: () => {},
});

// Popover wraps Trigger and Content
export function Popover({ children, open, onOpenChange }) {
  return (
    <PopoverContext.Provider value={{ open, onOpenChange }}>
      {children}
    </PopoverContext.Provider>
  );
}

// PopoverTrigger toggles the popover
export function PopoverTrigger({ children, asChild = false, ...props }) {
  const { open, onOpenChange } = React.useContext(PopoverContext);
  const child = React.Children.only(children);
  const triggerProps = {
    onClick: () => onOpenChange(!open),
    'aria-expanded': open,
    ...props,
  };
  if (asChild && React.isValidElement(child)) {
    return React.cloneElement(child, {
      ...triggerProps,
      ...child.props,
    });
  }
  return <button {...triggerProps}>{children}</button>;
}

// PopoverContent displays when open is true
export function PopoverContent({ children, className, style = {}, ...props }) {
  const { open } = React.useContext(PopoverContext);
  const combinedStyle = {
    display: open ? 'block' : 'none',
    ...style,
  };
  return (
    <div className={className} style={combinedStyle} {...props}>
      {children}
    </div>
  );
}
