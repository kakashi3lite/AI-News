import prisma from './db.js';
import { stripHtml, truncate, wordCount } from './utils.js';

/**
 * Summarization for the Market Signal dashboard.
 * - Extractive summaries are deterministic and work fully offline (accuracy first).
 * - Optionally enhanced with OpenAI when a key is present (guarded + non-blocking).
 * Summaries are persisted in ArticleSummary and served from cache thereafter.
 */

const STOPWORDS = new Set(
  'a an the and or but if then else for nor so yet to of in on at by with from up about into over after under again further then once here when where why how all any both each few more most other some such no not only own same too very just can will just should now today yesterday tomorrow says said new year month week day time people company says report amid after before during against between through during before above below between out off over under again further then once'.split(' ')
);

/** Deterministic extractive summary: top-scoring sentences by word frequency. */
export function extractiveSummary(content, maxSentences = 3, maxLength = 480) {
  const text = stripHtml(content || '');
  if (!text) return '';

  const sentences = text
    .split(/(?<=[.!?])\s+/)
    .map((s) => s.trim())
    .filter((s) => s.length > 25);

  if (sentences.length === 0) return truncate(text, maxLength);
  if (sentences.length <= maxSentences) return truncate(sentences.join(' '), maxLength);

  // Word frequency across the full text.
  const freq = {};
  text
    .toLowerCase()
    .match(/[a-z]+/g)
    ?.forEach((w) => {
      if (w.length > 3 && !STOPWORDS.has(w)) freq[w] = (freq[w] || 0) + 1;
    });

  const scored = sentences.map((sentence, i) => {
    let score = 0;
    sentence
      .toLowerCase()
      .match(/[a-z]+/g)
      ?.forEach((w) => {
        if (freq[w] && freq[w] > 1) score += freq[w];
      });
    // Favor early sentences slightly.
    score += 1 / (i + 1);
    return { sentence, score };
  });

  scored.sort((a, b) => b.score - a.score);
  const top = scored
    .slice(0, maxSentences)
    .sort((a, b) => sentences.indexOf(a.sentence) - sentences.indexOf(b.sentence))
    .map((x) => x.sentence);

  return truncate(top.join(' '), maxLength);
}

function shouldUseAI() {
  return Boolean(
    process.env.OPENAI_API_KEY &&
      process.env.USE_MOCK_DATA !== 'true' &&
      process.env.DISABLE_AI_SUMMARIES !== 'true'
  );
}

/** Direct OpenAI call (bypasses the legacy mock-prone client). */
async function callOpenAI(title, content) {
  const res = await fetch('https://api.openai.com/v1/chat/completions', {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${process.env.OPENAI_API_KEY}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      model: 'gpt-3.5-turbo',
      messages: [
        {
          role: 'system',
          content:
            'You are a neutral market analyst. Summarize the given news article in 2-3 factual sentences. Do not invent facts. If the article is not substantive, say so briefly.',
        },
        { role: 'user', content: `Title: ${title}\n\nArticle:\n${truncate(content, 4000)}` },
      ],
      max_tokens: 220,
      temperature: 0.2,
    }),
    signal: AbortSignal.timeout(15000),
  });
  if (!res.ok) throw new Error(`OpenAI HTTP ${res.status}`);
  const data = await res.json();
  return (data.choices?.[0]?.message?.content || '').trim();
}

/**
 * Ensure a standard summary exists for an article. Extractive by default;
 * AI-enhanced when a key is available. Never throws — falls back to extractive.
 */
export async function summarizeArticle(article, { withAI = true } = {}) {
  const existing = await prisma.articleSummary.findUnique({
    where: { articleId_summaryType: { articleId: article.id, summaryType: 'standard' } },
  });
  if (existing) return existing;

  const content = article.content || article.description || article.title || '';
  const extractive = extractiveSummary(content);
  if (!extractive) return null;

  let summaryText = extractive;
  let modelUsed = 'extractive';
  let confidence = 0.55;

  if (withAI && shouldUseAI()) {
    try {
      const ai = await callOpenAI(article.title, content);
      if (ai && ai.length > 20 && wordCount(ai) >= 15) {
        summaryText = ai;
        modelUsed = 'openai:gpt-3.5-turbo';
        confidence = 0.85;
      }
    } catch (err) {
      console.warn('[summarize] AI enhancement skipped:', err.message);
    }
  }

  try {
    return await prisma.articleSummary.create({
      data: {
        articleId: article.id,
        summaryType: 'standard',
        summaryText,
        modelUsed,
        confidence,
      },
    });
  } catch {
    return existing; // concurrent creation — already exists
  }
}
