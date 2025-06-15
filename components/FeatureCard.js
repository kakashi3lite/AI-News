import React from 'react';

/**
 * Simple card component for homepage features
 */
export default function FeatureCard({ icon, title, description }) {
  return (
    <div className="flex items-start p-4 bg-white rounded-lg shadow hover:shadow-md transition">
      <div className="text-blue-500 w-8 h-8 mr-4 flex-shrink-0">
        {icon}
      </div>
      <div className="flex-1">
        <h4 className="text-lg font-semibold mb-1">{title}</h4>
        <p className="text-gray-600 text-sm">{description}</p>
      </div>
    </div>
  );
}
