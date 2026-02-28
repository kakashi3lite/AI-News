/**
 * RSS/XML feed parsing module.
 * Handles fetching and parsing RSS feeds from various source formats.
 */

const axios = require('axios');
const { XMLParser } = require('fast-xml-parser');

const xmlParser = new XMLParser({
  ignoreAttributes: false,
  attributeNamePrefix: '@_'
});

const USER_AGENT = 'NewsForge-AI-Dashboard/1.0 (https://github.com/newsforge/ai-dashboard)';

const extractRSSItems = (parsed) => {
  if (parsed.rss?.channel?.item) {
    return Array.isArray(parsed.rss.channel.item) ? parsed.rss.channel.item : [parsed.rss.channel.item];
  }
  if (parsed.feed?.entry) {
    return Array.isArray(parsed.feed.entry) ? parsed.feed.entry : [parsed.feed.entry];
  }
  if (parsed.channel?.item) {
    return Array.isArray(parsed.channel.item) ? parsed.channel.item : [parsed.channel.item];
  }
  return [];
};

const fetchRSSFeed = async (url, timeout = 10000) => {
  const response = await axios.get(url, {
    timeout,
    headers: { 'User-Agent': USER_AGENT }
  });
  const parsed = xmlParser.parse(response.data);
  return extractRSSItems(parsed);
};

const fetchAPISource = async (source, options = {}) => {
  const params = { ...source.params };

  if (options.query) {
    params.q = options.query;
  }
  if (options.category && source.name === 'NewsAPI') {
    params.category = options.category;
  }

  const response = await axios.get(source.endpoint, {
    params: { ...params, key: source.key },
    timeout: 15000,
    headers: { 'User-Agent': 'NewsForge-AI-Dashboard/1.0' }
  });

  return response.data;
};

module.exports = { fetchRSSFeed, fetchAPISource, extractRSSItems };
