/**
 * Summarization configuration constants.
 * Defines configs, model settings, and prompt templates.
 */

const SUMMARIZATION_CONFIGS = {
  brief: { maxLength: 100, style: 'bullet-points', focus: 'key-facts' },
  standard: { maxLength: 250, style: 'paragraph', focus: 'comprehensive' },
  detailed: { maxLength: 500, style: 'structured', focus: 'analysis' },
  executive: { maxLength: 150, style: 'executive-summary', focus: 'business-impact' }
};

const MODEL_CONFIGS = {
  'o4-mini-high': { provider: 'o4', maxTokens: 4000, temperature: 0.3, capabilities: ['summarization', 'analysis', 'extraction'] },
  'gpt-3.5-turbo': { provider: 'openai', maxTokens: 4000, temperature: 0.3, capabilities: ['summarization', 'analysis', 'creative'] },
  'gpt-4': { provider: 'openai', maxTokens: 8000, temperature: 0.2, capabilities: ['advanced-analysis', 'reasoning', 'summarization'] }
};

const PROMPT_TEMPLATES = {
  extractive: {
    brief: "Extract the 3 most important facts from this article in bullet points:\n\n{content}\n\nKey Facts:",
    standard: "Summarize this article in 2-3 sentences focusing on the main points:\n\n{content}\n\nSummary:",
    detailed: "Provide a comprehensive summary of this article including context, main points, and implications:\n\n{content}\n\nDetailed Summary:"
  },
  abstractive: {
    brief: "Create a concise summary of this article in your own words (max 100 words):\n\n{content}\n\nSummary:",
    standard: "Write a clear, informative summary of this article (max 250 words):\n\n{content}\n\nSummary:",
    detailed: "Provide an in-depth analysis and summary of this article, including key insights and implications (max 500 words):\n\n{content}\n\nAnalysis:"
  },
  thematic: {
    brief: "Identify the main theme and 3 key points from this article:\n\n{content}\n\nTheme and Key Points:",
    standard: "Analyze the main themes and provide a structured summary:\n\n{content}\n\nThematic Summary:",
    detailed: "Provide a thematic analysis including main topics, subtopics, and their relationships:\n\n{content}\n\nThematic Analysis:"
  },
  sentiment: {
    brief: "Summarize this article and identify its overall sentiment (positive/negative/neutral):\n\n{content}\n\nSummary with Sentiment:",
    standard: "Provide a summary and detailed sentiment analysis of this article:\n\n{content}\n\nSentiment Analysis:",
    detailed: "Analyze the sentiment, tone, and emotional context of this article with a comprehensive summary:\n\n{content}\n\nDetailed Sentiment Analysis:"
  }
};

module.exports = { SUMMARIZATION_CONFIGS, MODEL_CONFIGS, PROMPT_TEMPLATES };
