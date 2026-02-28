/**
 * Topic classification module.
 * Classifies articles using keyword matching with optional transformer fallback.
 */

const path = require('path');
const fs = require('fs');
const { spawn } = require('child_process');
const { TOPIC_CATEGORIES } = require('./sources');

const classifyTopicsKeyword = (articles) => {
  console.log(`🏷️  Using keyword-based classification for ${articles.length} articles...`);

  return articles.map((article) => {
    const text = `${article.title} ${article.description}`.toLowerCase();
    const scores = {};

    Object.entries(TOPIC_CATEGORIES).forEach(([category, keywords]) => {
      scores[category] = keywords.reduce((score, keyword) => {
        const matches = (text.match(new RegExp(keyword, 'g')) || []).length;
        return score + matches;
      }, 0);
    });

    const bestCategory = Object.entries(scores).reduce(
      (best, [category, score]) => (score > best.score ? { category, score } : best),
      { category: 'general', score: 0 }
    );

    if (bestCategory.score > 0) {
      article.category = bestCategory.category;
      article.topicConfidence = bestCategory.score;
    }

    article.classificationMethod = 'keyword';
    return article;
  });
};

const basicTopicClassification = (articles) => {
  console.log(`🏷️  Using basic topic classification for ${articles.length} articles...`);

  return articles.map((article) => {
    const text = `${article.title} ${article.description}`.toLowerCase();

    if (text.includes('tech') || text.includes('ai') || text.includes('software')) {
      article.category = 'technology';
    } else if (text.includes('business') || text.includes('finance') || text.includes('market')) {
      article.category = 'business';
    } else if (text.includes('health') || text.includes('medical')) {
      article.category = 'health';
    } else if (text.includes('sport') || text.includes('game')) {
      article.category = 'sports';
    }

    return article;
  });
};

const classifyTopicsTransformer = async (articles, cacheDir) => {
  console.log(`🏷️  Classifying topics for ${articles.length} articles...`);

  const transformerPath = path.join(__dirname, 'transformer_classifier.py');
  if (!fs.existsSync(transformerPath)) {
    console.log('📝 Transformer classifier not found, using keyword classification');
    return classifyTopicsKeyword(articles);
  }

  console.log('🤖 Using transformer-based classification...');
  const fsPromises = require('fs').promises;

  const tempFile = path.join(cacheDir, `articles_${Date.now()}.json`);
  const articlesData = articles.map((a) => ({
    title: a.title,
    description: a.description,
    content: a.content,
    source: a.source,
    url: a.url
  }));
  await fsPromises.writeFile(tempFile, JSON.stringify(articlesData));

  const pythonScript = `
import asyncio, json, sys, os
sys.path.append('${__dirname.replace(/\\/g, '/')}')
from transformer_classifier import classify_topics_transformer
async def main():
    with open('${tempFile.replace(/\\/g, '/')}', 'r', encoding='utf-8') as f:
        articles = json.load(f)
    results = await classify_topics_transformer(articles)
    print(json.dumps(results))
    os.unlink('${tempFile.replace(/\\/g, '/')}')
asyncio.run(main())
  `;

  const pythonProcess = spawn('python', ['-c', pythonScript]);
  let output = '';
  let error = '';

  pythonProcess.stdout.on('data', (data) => { output += data.toString(); });
  pythonProcess.stderr.on('data', (data) => { error += data.toString(); });

  return new Promise((resolve) => {
    pythonProcess.on('close', async (code) => {
      try { await fsPromises.unlink(tempFile); } catch { /* already deleted */ }

      if (code === 0 && output.trim()) {
        try {
          const classified = JSON.parse(output.trim());
          console.log('✅ Transformer classification completed successfully');
          return resolve(classified);
        } catch {
          console.warn('⚠️ Failed to parse transformer results, falling back to keyword classification');
        }
      } else {
        console.warn(`⚠️ Transformer classification failed (code: ${code}), falling back`);
        if (error) console.warn('Error:', error);
      }

      resolve(classifyTopicsKeyword(articles));
    });
  });
};

module.exports = {
  classifyTopicsKeyword,
  basicTopicClassification,
  classifyTopicsTransformer
};
