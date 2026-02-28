/**
 * Theme Extraction Orchestrator
 * Coordinates keyword extraction, topic clustering, sentiment analysis, and trend detection.
 */

const crypto = require('crypto');
const fs = require('fs').promises;
const path = require('path');
const { extractKeywords } = require('./keywordAnalyzer');
const { analyzeSentiment, getDominantSentiment } = require('./sentimentAnalyzer');
const { detectTrends, generateInsights } = require('./trendDetector');

class ThemeExtractionEngine {
  constructor(options = {}) {
    this.options = {
      maxArticles: options.maxArticles || 100,
      maxThemes: options.maxThemes || 20,
      sentimentThreshold: options.sentimentThreshold || 0.1,
      outputDir: options.outputDir || './analytics/output',
      ...options
    };
    this.cache = new Map();
  }

  async extractThemes(articles) {
    console.log(`🔍 Starting theme extraction for ${articles.length} articles...`);

    const processedArticles = this.preprocessArticles(articles);
    const keywords = extractKeywords(processedArticles);
    const topics = this.clusterTopics(keywords, processedArticles);
    const sentimentAnalysis = analyzeSentiment(processedArticles, topics, this.options.sentimentThreshold);
    const themes = this.calculateThemeScores(topics, sentimentAnalysis);
    const trends = await detectTrends(themes, this.options.outputDir);
    const insights = generateInsights(themes, trends, sentimentAnalysis);

    const result = {
      timestamp: new Date().toISOString(),
      articleCount: articles.length,
      themes,
      trends,
      sentiment: sentimentAnalysis,
      insights,
      metadata: { processingTime: Date.now(), version: '1.0.0', engine: 'ThemeExtractionEngine' }
    };

    await this.saveResults(result);
    console.log(`✅ Theme extraction completed. Found ${themes.length} themes, ${trends.length} trends`);
    return result;
  }

  preprocessArticles(articles) {
    return articles
      .map((article) => {
        const text = `${article.title || ''} ${article.description || ''} ${article.content || ''}`;
        return {
          id: article.id || crypto.randomUUID(),
          title: article.title || '',
          content: text,
          source: article.source || 'unknown',
          publishedAt: article.publishedAt || new Date().toISOString(),
          category: article.category || 'general',
          url: article.url || '',
          cleanText: text.toLowerCase().replace(/[^a-zA-Z0-9\s]/g, ' ').replace(/\s+/g, ' ').trim(),
          wordCount: text.split(/\s+/).length,
          sentences: text.split(/[.!?]+/).filter((s) => s.trim().length > 0)
        };
      })
      .filter((article) => article.wordCount >= 10);
  }

  clusterTopics(keywords, articles) {
    const categoryGroups = {};
    keywords.forEach((kw) => {
      if (!categoryGroups[kw.category]) categoryGroups[kw.category] = [];
      categoryGroups[kw.category].push(kw);
    });

    const topics = [];
    Object.entries(categoryGroups).forEach(([category, categoryKeywords]) => {
      if (categoryKeywords.length < 2) return;

      const topicScore = categoryKeywords.reduce((sum, kw) => sum + kw.tfidf, 0);
      const keywordSet = new Set(categoryKeywords.map((kw) => kw.word));
      const relatedArticles = articles
        .map((a) => {
          const words = a.cleanText.split(/\s+/);
          const matchCount = words.filter((w) => keywordSet.has(w)).length;
          return { ...a, relevanceScore: matchCount / words.length, matchCount };
        })
        .filter((a) => a.matchCount >= 2)
        .sort((a, b) => b.relevanceScore - a.relevanceScore);

      topics.push({
        name: category,
        keywords: categoryKeywords.slice(0, 10),
        score: topicScore,
        articleCount: relatedArticles.length,
        articles: relatedArticles.slice(0, 5),
        trend: 'stable',
        sentiment: 'neutral'
      });
    });

    return topics.sort((a, b) => b.score - a.score).slice(0, this.options.maxThemes);
  }

  calculateThemeScores(topics, sentimentAnalysis) {
    return topics
      .map((topic) => {
        const sentimentData = sentimentAnalysis.byTopic[topic.name] || { positive: 33.3, negative: 33.3, neutral: 33.3 };
        const volumeScore = Math.log(topic.articleCount + 1) * 10;
        const sentimentScore = parseFloat(sentimentData.positive) - parseFloat(sentimentData.negative);
        const compositeScore = Math.round((topic.score * 0.4 + volumeScore * 0.4 + sentimentScore * 0.2) * 100) / 100;
        return { ...topic, sentiment: getDominantSentiment(sentimentData), sentimentDistribution: sentimentData, compositeScore, rank: 0 };
      })
      .sort((a, b) => b.compositeScore - a.compositeScore)
      .map((theme, index) => ({ ...theme, rank: index + 1 }));
  }

  async saveResults(results) {
    try {
      await fs.mkdir(this.options.outputDir, { recursive: true });
      const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
      await fs.writeFile(path.join(this.options.outputDir, `themes_${timestamp}.json`), JSON.stringify(results, null, 2));
      await fs.writeFile(path.join(this.options.outputDir, 'latest_themes.json'), JSON.stringify(results, null, 2));
      await this.updateThemeHistory(results.themes);
    } catch (error) {
      console.error('❌ Failed to save results:', error);
    }
  }

  async updateThemeHistory(themes) {
    try {
      const historyFile = path.join(this.options.outputDir, 'theme_history.json');
      let history = { themes: [] };
      try { history = JSON.parse(await fs.readFile(historyFile, 'utf8')); } catch { /* start fresh */ }

      history.themes = themes.map((t) => ({ name: t.name, compositeScore: t.compositeScore, articleCount: t.articleCount, timestamp: new Date().toISOString() }));
      const cutoff = new Date(Date.now() - 24 * 60 * 60 * 1000);
      history.themes = history.themes.filter((t) => new Date(t.timestamp) > cutoff);
      await fs.writeFile(historyFile, JSON.stringify(history, null, 2));
    } catch (error) {
      console.error('❌ Failed to update theme history:', error);
    }
  }

  async getCurrentThemes() {
    const data = await fs.readFile(path.join(this.options.outputDir, 'latest_themes.json'), 'utf8');
    return JSON.parse(data);
  }

  async getThemeHistory(hours = 24) {
    try {
      const data = await fs.readFile(path.join(this.options.outputDir, 'theme_history.json'), 'utf8');
      const cutoff = new Date(Date.now() - hours * 60 * 60 * 1000);
      return JSON.parse(data).themes.filter((t) => new Date(t.timestamp) > cutoff);
    } catch { return []; }
  }
}

class ThemeScheduler {
  constructor(engine, newsIngestor) {
    this.engine = engine;
    this.newsIngestor = newsIngestor;
    this.isRunning = false;
    this.intervalId = null;
  }

  start() {
    if (this.isRunning) return;
    this.isRunning = true;
    this.runExtractionCycle();
    this.intervalId = setInterval(() => this.runExtractionCycle(), 60 * 60 * 1000);
  }

  stop() {
    if (this.intervalId) clearInterval(this.intervalId);
    this.intervalId = null;
    this.isRunning = false;
  }

  async runExtractionCycle() {
    try {
      const articles = await this.newsIngestor.getTopArticles(100);
      if (articles.length === 0) return;
      await this.engine.extractThemes(articles);
    } catch (error) {
      console.error('❌ Theme extraction cycle failed:', error);
    }
  }
}

module.exports = { ThemeExtractionEngine, ThemeScheduler };

if (require.main === module) {
  const engine = new ThemeExtractionEngine();
  const mockArticles = [
    { id: '1', title: 'AI Breakthrough in Machine Learning', content: 'Researchers have achieved a significant breakthrough in artificial intelligence and machine learning technology...', source: 'TechNews', publishedAt: new Date().toISOString() },
    { id: '2', title: 'Climate Change Policy Updates', content: 'Government announces new environmental policies to address climate change and reduce carbon emissions...', source: 'EnviroDaily', publishedAt: new Date().toISOString() },
    { id: '3', title: 'Stock Market Analysis', content: 'Financial markets show positive trends as technology stocks continue to rise amid economic recovery...', source: 'FinanceToday', publishedAt: new Date().toISOString() }
  ];
  engine.extractThemes(mockArticles).then((results) => {
    console.log(`\n📊 Themes: ${results.themes.length}, Trends: ${results.trends.length}`);
    results.themes.slice(0, 5).forEach((t, i) => console.log(`${i + 1}. ${t.name} (Score: ${t.compositeScore})`));
  }).catch(console.error);
}
