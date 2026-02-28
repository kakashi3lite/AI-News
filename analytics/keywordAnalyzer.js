/**
 * Keyword extraction and TF-IDF analysis module.
 */

const { STOP_WORDS, THEME_CATEGORIES } = require('./themeConfig');

const categorizeKeyword = (word) => {
  for (const [category, keywords] of Object.entries(THEME_CATEGORIES)) {
    if (keywords.some((keyword) => word.includes(keyword) || keyword.includes(word))) {
      return category;
    }
  }
  return 'general';
};

const extractKeywords = (articles) => {
  const wordFreq = new Map();
  const documentFreq = new Map();
  const totalDocs = articles.length;

  articles.forEach((article) => {
    const words = article.cleanText.split(/\s+/);
    const uniqueWords = new Set(words);

    words.forEach((word) => {
      if (word.length >= 3 && !STOP_WORDS.has(word)) {
        wordFreq.set(word, (wordFreq.get(word) || 0) + 1);
      }
    });

    uniqueWords.forEach((word) => {
      if (word.length >= 3 && !STOP_WORDS.has(word)) {
        documentFreq.set(word, (documentFreq.get(word) || 0) + 1);
      }
    });
  });

  const keywords = [];
  wordFreq.forEach((tf, word) => {
    const df = documentFreq.get(word) || 1;
    const idf = Math.log(totalDocs / df);
    const tfidf = tf * idf;

    if (tfidf > 1.0) {
      keywords.push({ word, frequency: tf, documentFrequency: df, tfidf, category: categorizeKeyword(word) });
    }
  });

  return keywords.sort((a, b) => b.tfidf - a.tfidf).slice(0, 200);
};

module.exports = { extractKeywords, categorizeKeyword };
