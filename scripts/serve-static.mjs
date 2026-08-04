// Tiny static server that mirrors GitHub Pages sub-path hosting.
// Serves ./out and maps /AI-News/* → out/* (like https://user.github.io/AI-News).
// Usage: node scripts/serve-static.mjs [basePath] [port]
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';

const root = path.join(process.cwd(), 'out');
const base = process.argv[2] || '/AI-News';
const port = Number(process.argv[3] || 8080);

const MIME = {
  '.html': 'text/html',
  '.js': 'text/javascript',
  '.css': 'text/css',
  '.json': 'application/json',
  '.svg': 'image/svg+xml',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.webp': 'image/webp',
  '.ico': 'image/x-icon',
  '.txt': 'text/plain',
  '.map': 'application/json',
};

http
  .createServer((req, res) => {
    let url = decodeURIComponent(new URL(req.url, 'http://x').pathname);
    if (url.startsWith(base)) url = url.slice(base.length) || '/';

    let filePath = path.join(root, url);
    try {
      if (fs.existsSync(filePath) && fs.statSync(filePath).isDirectory()) {
        filePath = path.join(filePath, 'index.html');
      }
      if (!fs.existsSync(filePath)) filePath = path.join(root, `${url}.html`);
      if (!fs.existsSync(filePath)) filePath = path.join(root, '404.html');
      const ext = path.extname(filePath);
      res.writeHead(200, { 'Content-Type': MIME[ext] || 'application/octet-stream' });
      fs.createReadStream(filePath).pipe(res);
    } catch {
      res.writeHead(404, { 'Content-Type': 'text/plain' });
      res.end('Not found');
    }
  })
  .listen(port, () => {
    console.log(`Static server: http://localhost:${port}${base}  (root: ${root})`);
  });
