import axios from 'axios';

// Mock summaries for development/testing when API is unavailable
const MOCK_SUMMARIES = [
  "This comprehensive analysis examines current market trends and their implications for future business strategies. The report highlights key performance indicators and provides actionable insights for stakeholders.",
  "Recent policy developments have created new opportunities and challenges across multiple sectors. This summary outlines the main changes and their expected impact on industry operations.",
  "Scientific research continues to advance our understanding of complex global issues. This article summarizes breakthrough findings and their potential applications in real-world scenarios.",
  "Technology innovation drives transformation across industries. This piece explores emerging trends, adoption patterns, and the competitive landscape shaping the digital future.",
  "Economic indicators suggest shifting patterns in consumer behavior and market dynamics. The analysis provides insights into current conditions and forecasts for upcoming quarters.",
  "Environmental sustainability initiatives gain momentum as organizations implement new strategies. This summary covers recent developments and their effectiveness in achieving climate goals."
];

/**
 * Calls OpenAI Chat Completion API for summarization.
 * Docs: https://axios-http.com/docs/api_intro
 */
export async function queryOpenAI(prompt) {
  const apiKey = process.env.OPENAI_API_KEY;
  
  // Check if we're in development mode or missing API config
  const isDevelopment = process.env.NODE_ENV === 'development';
  const isMockMode = process.env.USE_MOCK_DATA === 'true';
  
  if (!apiKey || isMockMode) {
    console.log('🔧 Using mock OpenAI response (API not configured or mock mode enabled)');
    // Return a random mock summary
    const randomIndex = Math.floor(Math.random() * MOCK_SUMMARIES.length);
    return MOCK_SUMMARIES[randomIndex];
  }
  
  try {
    const response = await axios.post(
      'https://api.openai.com/v1/chat/completions',
      {
        model: 'gpt-3.5-turbo',
        messages: [
          { role: 'system', content: 'Provide clear, concise summaries.' },
          { role: 'user', content: prompt }
        ],
        max_tokens: 256,
        temperature: 0.7
      },
      {
        headers: {
          'Authorization': `Bearer ${apiKey}`,
          'Content-Type': 'application/json'
        },
        timeout: 15000 // 15 second timeout
      }
    );
    return response.data.choices?.[0]?.message.content || '';
  } catch (error) {
    console.warn('⚠️ OpenAI API failed, falling back to mock data:', error.message);
    // Fallback to mock data on API failure
    const randomIndex = Math.floor(Math.random() * MOCK_SUMMARIES.length);
    return MOCK_SUMMARIES[randomIndex];
  }
}
