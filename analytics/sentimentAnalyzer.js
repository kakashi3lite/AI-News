/**
 * Sentiment analysis module.
 * Calculates sentiment scores for articles and topics.
 */

const { SENTIMENT_KEYWORDS } = require('./themeConfig');

const calculateSentiment = (text, threshold = 0.1) => {
  const words = text.toLowerCase().split(/\s+/);
  let positiveScore = 0;
  let negativeScore = 0;

  words.forEach((word) => {
    if (SENTIMENT_KEYWORDS.positive.some((pos) => word.includes(pos))) positiveScore++;
    if (SENTIMENT_KEYWORDS.negative.some((neg) => word.includes(neg))) negativeScore++;
  });

  const totalScore = positiveScore + negativeScore;
  if (totalScore === 0) return { label: 'neutral', score: 0 };

  const sentimentScore = (positiveScore - negativeScore) / totalScore;
  if (sentimentScore > threshold) return { label: 'positive', score: sentimentScore };
  if (sentimentScore < -threshold) return { label: 'negative', score: sentimentScore };
  return { label: 'neutral', score: sentimentScore };
};

const getDominantSentiment = (sentimentData) => {
  const positive = parseFloat(sentimentData.positive);
  const negative = parseFloat(sentimentData.negative);
  const neutral = parseFloat(sentimentData.neutral);
  if (positive > negative && positive > neutral) return 'positive';
  if (negative > positive && negative > neutral) return 'negative';
  return 'neutral';
};

const analyzeSentiment = (articles, topics, threshold = 0.1) => {
  const overallSentiment = { positive: 0, negative: 0, neutral: 0 };
  const topicSentiments = {};

  articles.forEach((article) => {
    const sentiment = calculateSentiment(article.content, threshold);
    overallSentiment[sentiment.label]++;

    topics.forEach((topic) => {
      const isRelated = topic.keywords.some((kw) => article.cleanText.includes(kw.word));
      if (!isRelated) return;

      if (!topicSentiments[topic.name]) {
        topicSentiments[topic.name] = { positive: 0, negative: 0, neutral: 0 };
      }
      topicSentiments[topic.name][sentiment.label]++;
    });
  });

  const total = articles.length;
  const sentimentAnalysis = {
    overall: {
      positive: (overallSentiment.positive / total * 100).toFixed(1),
      negative: (overallSentiment.negative / total * 100).toFixed(1),
      neutral: (overallSentiment.neutral / total * 100).toFixed(1)
    },
    byTopic: {}
  };

  Object.entries(topicSentiments).forEach(([topic, sentiment]) => {
    const topicTotal = sentiment.positive + sentiment.negative + sentiment.neutral;
    if (topicTotal === 0) return;
    sentimentAnalysis.byTopic[topic] = {
      positive: (sentiment.positive / topicTotal * 100).toFixed(1),
      negative: (sentiment.negative / topicTotal * 100).toFixed(1),
      neutral: (sentiment.neutral / topicTotal * 100).toFixed(1)
    };
  });

  return sentimentAnalysis;
};

module.exports = { calculateSentiment, getDominantSentiment, analyzeSentiment };
