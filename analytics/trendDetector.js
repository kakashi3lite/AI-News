/**
 * Trend detection module.
 * Compares current themes against historical data to identify trends.
 */

const fs = require('fs').promises;
const path = require('path');

const loadHistoricalData = async (outputDir) => {
  try {
    const historyFile = path.join(outputDir, 'theme_history.json');
    const data = await fs.readFile(historyFile, 'utf8');
    return JSON.parse(data).themes || [];
  } catch {
    return [];
  }
};

const detectTrends = async (themes, outputDir) => {
  const historicalData = await loadHistoricalData(outputDir);
  const trends = [];

  themes.forEach((theme) => {
    const historical = historicalData.find((h) => h.name === theme.name);
    let trendDirection = 'new';
    let velocityScore = 0;

    if (historical) {
      const scoreDiff = theme.compositeScore - historical.compositeScore;
      const articleDiff = theme.articleCount - historical.articleCount;
      velocityScore = (scoreDiff + articleDiff) / 2;

      if (velocityScore > 5) trendDirection = 'rising';
      else if (velocityScore < -5) trendDirection = 'falling';
      else trendDirection = 'stable';
    }

    if (trendDirection !== 'rising' && trendDirection !== 'new') return;

    trends.push({
      theme: theme.name,
      direction: trendDirection,
      velocity: velocityScore,
      currentScore: theme.compositeScore,
      previousScore: historical?.compositeScore || 0,
      articleCount: theme.articleCount,
      keywords: theme.keywords.slice(0, 5).map((kw) => kw.word),
      sentiment: theme.sentiment
    });
  });

  return trends.sort((a, b) => b.velocity - a.velocity);
};

const generateInsights = (themes, trends, sentimentAnalysis) => {
  const topTheme = themes[0];
  const trendingCount = trends.filter((t) => t.direction === 'rising').length;

  const insights = {
    summary: `Analysis of ${themes.length} themes reveals "${topTheme.name}" as the dominant topic with ${topTheme.articleCount} articles. ${trendingCount} themes are currently trending upward.`,
    keyFindings: [
      `Top theme: ${topTheme.name} (${topTheme.compositeScore} score)`,
      `Overall sentiment: ${sentimentAnalysis.overall.positive}% positive, ${sentimentAnalysis.overall.negative}% negative`,
      `Trending topics: ${trends.slice(0, 3).map((t) => t.theme).join(', ')}`,
      `Most discussed categories: ${themes.slice(0, 5).map((t) => t.name).join(', ')}`
    ],
    recommendations: [],
    alerts: []
  };

  if (trends.length > 0) insights.recommendations.push(`Monitor trending topic: ${trends[0].theme}`);

  const negativeThemes = themes.filter((t) => t.sentiment === 'negative');
  if (negativeThemes.length > 0) insights.recommendations.push(`Address negative sentiment in: ${negativeThemes[0].name}`);

  trends.forEach((trend) => {
    if (trend.velocity > 20) {
      insights.alerts.push({ type: 'high_velocity', message: `Rapid growth detected in ${trend.theme} (+${trend.velocity.toFixed(1)})`, severity: 'high' });
    }
  });

  Object.entries(sentimentAnalysis.byTopic)
    .filter(([, sentiment]) => parseFloat(sentiment.negative) > 60)
    .forEach(([topic]) => {
      insights.alerts.push({ type: 'negative_sentiment', message: `High negative sentiment detected in ${topic}`, severity: 'medium' });
    });

  return insights;
};

module.exports = { detectTrends, generateInsights, loadHistoricalData };
