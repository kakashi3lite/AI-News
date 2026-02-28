/**
 * Article normalization module.
 * Converts raw feed items into a standard article format.
 */

const crypto = require('crypto');

const cleanText = (text) => {
  if (!text) return '';
  return text.replace(/<[^>]*>/g, '').replace(/\s+/g, ' ').trim();
};

const normalizeDate = (dateStr) => {
  try {
    return new Date(dateStr).toISOString();
  } catch {
    return new Date().toISOString();
  }
};

const generateArticleId = (title, url, source) => {
  const content = `${title}:${url}:${source}:${Date.now()}`;
  return crypto.createHash('sha256').update(content).digest('hex').slice(0, 16);
};

const extractImageUrl = (item) => {
  if (item.enclosure?.['@_url']) return item.enclosure['@_url'];
  if (item['media:content']?.['@_url']) return item['media:content']['@_url'];
  if (item.image?.url) return item.image.url;
  return '';
};

const extractTags = (item) => {
  if (!item.category) return [];
  const cats = Array.isArray(item.category) ? item.category : [item.category];
  return cats.map((c) => (typeof c === 'string' ? c : c['#text'] || c._ || '')).filter(Boolean);
};

const normalizeRSSArticle = (item, source) => {
  const title = item.title || item.summary || 'No title';
  const description = item.description || item.summary || item.content || '';
  const link = item.link?.['@_href'] || item.link || item.guid || '';
  const pubDate = item.pubDate || item.published || item.updated || new Date().toISOString();

  return {
    id: generateArticleId(title, link, source.name),
    title: cleanText(title),
    description: cleanText(description),
    content: cleanText(description),
    url: link,
    image: extractImageUrl(item),
    publishedAt: normalizeDate(pubDate),
    source: { name: source.name, url: source.url },
    category: source.category || 'general',
    tags: extractTags(item),
    ingestionMethod: 'rss',
    fetchedAt: new Date().toISOString()
  };
};

const normalizeNewsAPIArticle = (item, source) => ({
  id: generateArticleId(item.title, item.url, source.name),
  title: cleanText(item.title || 'No title'),
  description: cleanText(item.description || ''),
  content: cleanText(item.content || item.description || ''),
  url: item.url || '',
  image: item.urlToImage || '',
  publishedAt: normalizeDate(item.publishedAt),
  source: { name: item.source?.name || source.name, url: item.url },
  category: 'general',
  tags: [],
  ingestionMethod: 'api-newsapi',
  fetchedAt: new Date().toISOString()
});

const normalizeGoogleNewsArticle = (item, source) => ({
  id: generateArticleId(item.title, item.link, source.name),
  title: cleanText(item.title || 'No title'),
  description: cleanText(item.snippet || ''),
  content: cleanText(item.snippet || ''),
  url: item.link || '',
  image: item.pagemap?.cse_image?.[0]?.src || '',
  publishedAt: new Date().toISOString(),
  source: { name: item.displayLink || source.name, url: item.link },
  category: 'general',
  tags: item.pagemap?.metatags?.[0]?.news_keywords?.split(',').map((t) => t.trim()).filter(Boolean) || [],
  ingestionMethod: 'api-google',
  fetchedAt: new Date().toISOString()
});

module.exports = {
  cleanText,
  normalizeDate,
  generateArticleId,
  normalizeRSSArticle,
  normalizeNewsAPIArticle,
  normalizeGoogleNewsArticle
};
