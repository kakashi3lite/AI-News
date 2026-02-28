/**
 * Summary quality analysis module.
 * Calculates readability, keyword extraction, and quality scores.
 */

const { SUMMARIZATION_CONFIGS } = require('./summarizationConfig');
const { countSyllables } = require('./contentPreprocessor');

const calculateReadability = (text) => {
  const sentences = text.split(/[.!?]+/).filter((s) => s.trim().length > 0);
  const words = text.split(/\s+/).filter((w) => w.length > 0);
  const syllables = words.reduce((count, word) => count + countSyllables(word), 0);

  if (sentences.length === 0 || words.length === 0) return 0;

  const score = 206.835 - 1.015 * (words.length / sentences.length) - 84.6 * (syllables / words.length);
  return Math.max(0, Math.min(100, Math.round(score)));
};

const extractKeywords = (text) => {
  const words = text
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, '')
    .split(/\s+/)
    .filter((word) => word.length > 3);

  const frequency = {};
  words.forEach((word) => {
    frequency[word] = (frequency[word] || 0) + 1;
  });

  return Object.entries(frequency)
    .sort(([, a], [, b]) => b - a)
    .slice(0, 10)
    .map(([word]) => word);
};

const calculateQualityScore = (summary, config) => {
  let score = 50;

  const targetLength = SUMMARIZATION_CONFIGS[config.type]?.maxLength || 250;
  const lengthRatio = summary.wordCount / targetLength;
  if (lengthRatio >= 0.7 && lengthRatio <= 1.2) score += 20;
  else if (lengthRatio >= 0.5 && lengthRatio <= 1.5) score += 10;

  if (summary.readability >= 60) score += 15;
  else if (summary.readability >= 40) score += 10;

  if (summary.keywords?.length >= 5) score += 10;
  if (summary.keyPoints?.length >= 3) score += 5;

  return Math.min(100, Math.max(0, score));
};

const enhanceSummary = (summary, config) => ({
  ...summary,
  readability: calculateReadability(summary.text),
  keywords: extractKeywords(summary.text),
  qualityScore: calculateQualityScore(
    { ...summary, readability: calculateReadability(summary.text), keywords: extractKeywords(summary.text) },
    config
  )
});

module.exports = { calculateReadability, extractKeywords, calculateQualityScore, enhanceSummary };
