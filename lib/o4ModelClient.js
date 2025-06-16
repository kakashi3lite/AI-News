import axios from 'axios';
// Docs: https://axios-http.com/docs/api_intro

// Mock summaries for development/testing when API is unavailable
const MOCK_SUMMARIES = [
  "This article discusses recent developments in technology and their impact on society. Key points include innovation trends, market analysis, and future implications for various industries.",
  "The report covers economic indicators and market performance. Analysis shows mixed signals with some sectors showing growth while others face challenges in the current environment.",
  "Breaking news update on political developments. The article examines policy changes, stakeholder reactions, and potential consequences for upcoming legislative sessions.",
  "Health and science news focusing on recent research findings. The study reveals important insights about public health measures and their effectiveness in current conditions.",
  "Environmental update covering climate change initiatives and sustainability efforts. The piece highlights progress made and challenges that remain in achieving environmental goals."
];

// Utility to call o4-mini-high model for search/summarization
// Set O4_MODEL_API_KEY and O4_MODEL_API_URL in .env.local
export async function queryO4Model(prompt) {
  const apiKey = process.env.O4_MODEL_API_KEY;
  const apiUrl = process.env.O4_MODEL_API_URL;
  
  // Check if we're in development mode or missing API config
  const isDevelopment = process.env.NODE_ENV === 'development';
  const isMockMode = process.env.USE_MOCK_DATA === 'true';
  
  if (!apiKey || !apiUrl || isMockMode) {
    console.log('🔧 Using mock O4 model response (API not configured or mock mode enabled)');
    // Return a random mock summary
    const randomIndex = Math.floor(Math.random() * MOCK_SUMMARIES.length);
    return MOCK_SUMMARIES[randomIndex];
  }

  try {
    const res = await axios.post(apiUrl, {
      prompt,
      model: 'o4-mini-high',
      max_tokens: 256,
    }, {
      headers: {
        'Authorization': `Bearer ${apiKey}`,
        'Content-Type': 'application/json',
      },
      timeout: 10000 // 10 second timeout
    });

    return res.data.choices?.[0]?.text || '';
  } catch (error) {
    console.warn('⚠️ O4 model API failed, falling back to mock data:', error.message);
    // Fallback to mock data on API failure
    const randomIndex = Math.floor(Math.random() * MOCK_SUMMARIES.length);
    return MOCK_SUMMARIES[randomIndex];
  }
}
