/**
 * Content preprocessing module.
 * Cleans, validates, and prepares text for summarization.
 */

const preprocessContent = (content) => {
  if (!content || typeof content !== 'string') return null;

  let cleaned = content.replace(/<[^>]*>/g, '');
  cleaned = cleaned.replace(/\s+/g, ' ').trim();

  if (cleaned.length < 50) return null;
  if (cleaned.length > 10000) cleaned = cleaned.substring(0, 10000) + '...';

  return cleaned;
};

const extractArticleContent = (article) => {
  const content = article.content || article.description || article.title || '';
  if (!content.trim()) throw new Error('Article has no content to summarize');
  return content;
};

const countWords = (text) =>
  text.trim().split(/\s+/).filter((word) => word.length > 0).length;

const countSyllables = (word) => {
  word = word.toLowerCase();
  if (word.length <= 3) return 1;
  const vowels = word.match(/[aeiouy]+/g);
  let count = vowels ? vowels.length : 1;
  if (word.endsWith('e')) count--;
  return count || 1;
};

module.exports = { preprocessContent, extractArticleContent, countWords, countSyllables };
