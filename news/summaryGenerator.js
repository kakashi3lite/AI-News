/**
 * Summary generation module.
 * Handles prompt building and model invocation for summarization.
 */

const { queryO4Model } = require('../lib/o4ModelClient');
const { queryOpenAI } = require('../lib/openaiClient');
const { MODEL_CONFIGS, PROMPT_TEMPLATES, SUMMARIZATION_CONFIGS } = require('./summarizationConfig');

const buildPrompt = (template, content, config) => {
  let prompt = template.replace('{content}', content);

  if (config.title) prompt = `Article Title: ${config.title}\n\n${prompt}`;
  if (config.category) prompt = `Category: ${config.category}\n${prompt}`;

  const summaryConfig = SUMMARIZATION_CONFIGS[config.type];
  if (summaryConfig) {
    prompt += `\n\nPlease limit the summary to approximately ${summaryConfig.maxLength} words.`;
  }

  return prompt;
};

const extractBulletPoints = (text) => {
  const points = text
    .split('\n')
    .map((line) => line.trim())
    .filter((line) => line.match(/^[•\-*]\s+/) || line.match(/^\d+\.\s+/))
    .map((line) => line.replace(/^[•\-*\d.\s]+/, ''));
  return points.length > 0 ? points : null;
};

const extractSentiment = (text) => {
  const lower = text.toLowerCase();
  const positiveWords = ['positive', 'optimistic', 'good', 'success', 'growth', 'improvement'];
  const negativeWords = ['negative', 'pessimistic', 'bad', 'failure', 'decline', 'crisis'];
  const pos = positiveWords.reduce((c, w) => c + (lower.includes(w) ? 1 : 0), 0);
  const neg = negativeWords.reduce((c, w) => c + (lower.includes(w) ? 1 : 0), 0);
  if (pos > neg) return 'positive';
  if (neg > pos) return 'negative';
  return 'neutral';
};

const extractThemes = (text) => {
  const themes = text
    .split('\n')
    .filter((line) => line.includes('Theme:') || line.includes('Topic:'))
    .map((line) => line.split(':')[1]?.trim())
    .filter(Boolean);
  return themes.length > 0 ? themes : null;
};

const parseSummaryResponse = (response, config) => {
  const summary = {
    text: response.trim(),
    type: config.type,
    style: config.style,
    model: config.model,
    wordCount: response.trim().split(/\s+/).filter((w) => w.length > 0).length,
    generatedAt: new Date().toISOString()
  };

  if (config.style === 'extractive' && config.type === 'brief') {
    summary.keyPoints = extractBulletPoints(response);
  }
  if (config.style === 'sentiment') summary.sentiment = extractSentiment(response);
  if (config.style === 'thematic') summary.themes = extractThemes(response);

  return summary;
};

const generateSummary = async (content, config) => {
  const modelConfig = MODEL_CONFIGS[config.model];
  if (!modelConfig) throw new Error(`Unsupported model: ${config.model}`);

  const promptTemplate = PROMPT_TEMPLATES[config.style]?.[config.type] || PROMPT_TEMPLATES.abstractive[config.type];
  if (!promptTemplate) throw new Error(`No prompt template found for ${config.style}/${config.type}`);

  const prompt = buildPrompt(promptTemplate, content, config);

  if (modelConfig.provider === 'o4') {
    const response = await queryO4Model(prompt);
    return parseSummaryResponse(response, config);
  }
  if (modelConfig.provider === 'openai') {
    const response = await queryOpenAI(prompt);
    return parseSummaryResponse(response, config);
  }

  throw new Error(`Unsupported model provider: ${modelConfig.provider}`);
};

module.exports = { generateSummary, buildPrompt, parseSummaryResponse };
