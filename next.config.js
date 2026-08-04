/** @type {import('next').NextConfig} */

const basePath = process.env.NEXT_PUBLIC_BASE_PATH || '';
const isStatic = process.env.NEXT_PUBLIC_STATIC === 'true';

const nextConfig = {
  reactStrictMode: true,
  // Static export mode (GitHub Pages / Netlify / Vercel static).
  ...(isStatic ? { output: 'export', distDir: 'out', trailingSlash: true } : {}),
  // Sub-path hosting (e.g. GitHub Pages: https://user.github.io/AI-News).
  ...(basePath ? { basePath } : {}),
  images: { unoptimized: true },
  async headers() {
    return [
      {
        source: '/:path*',
        headers: [
          { key: 'X-Frame-Options', value: 'DENY' },
          { key: 'X-Content-Type-Options', value: 'nosniff' },
          { key: 'Referrer-Policy', value: 'strict-origin-when-cross-origin' },
          { key: 'Permissions-Policy', value: 'camera=(), microphone=(), geolocation=()' },
          { key: 'X-DNS-Prefetch-Control', value: 'on' },
        ],
      },
    ];
  },
};

module.exports = nextConfig;
