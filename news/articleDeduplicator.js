/**
 * Article deduplication module.
 * Removes duplicate articles based on title fingerprinting.
 */

const crypto = require('crypto');

const generateFingerprint = (article) => {
  const titleWords = article.title
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, '')
    .split(/\s+/)
    .filter((word) => word.length > 3)
    .slice(0, 5)
    .sort()
    .join(' ');

  let urlDomain = '';
  if (article.url) {
    try {
      urlDomain = new URL(article.url).hostname;
    } catch {
      urlDomain = article.url.replace(/[^a-zA-Z0-9.-]/g, '').slice(0, 50);
    }
  }

  return crypto.createHash('md5').update(`${titleWords}:${urlDomain}`).digest('hex');
};

const deduplicateArticles = (articles) => {
  console.log(`🔄 Deduplicating ${articles.length} articles...`);

  const seen = new Set();
  const unique = [];

  for (const article of articles) {
    const fingerprint = generateFingerprint(article);
    if (seen.has(fingerprint)) continue;
    seen.add(fingerprint);
    unique.push(article);
  }

  console.log(`✅ Removed ${articles.length - unique.length} duplicates`);
  return unique;
};

module.exports = { deduplicateArticles, generateFingerprint };
