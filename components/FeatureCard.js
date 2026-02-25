import React from 'react';

export default function FeatureCard({ icon, title, description }) {
  return (
    <div
      className="flex flex-col p-5 rounded-xl border transition-all duration-200 hover:-translate-y-0.5 group cursor-pointer"
      style={{
        backgroundColor: 'var(--surface)',
        borderColor: 'var(--border)',
        boxShadow: 'var(--shadow-card)',
      }}
      onMouseEnter={e => { e.currentTarget.style.boxShadow = 'var(--shadow-card-hover)'; }}
      onMouseLeave={e => { e.currentTarget.style.boxShadow = 'var(--shadow-card)'; }}
    >
      <div
        className="w-10 h-10 rounded-lg mb-4 flex items-center justify-center flex-shrink-0 transition-colors duration-200"
        style={{ backgroundColor: 'var(--accent)', color: 'var(--accent-fg)' }}
      >
        {icon}
      </div>
      <h4
        className="font-display font-semibold text-base mb-1.5"
        style={{ color: 'var(--fg)' }}
      >
        {title}
      </h4>
      <p
        className="text-sm leading-relaxed"
        style={{ color: 'var(--fg-muted)' }}
      >
        {description}
      </p>
    </div>
  );
}
