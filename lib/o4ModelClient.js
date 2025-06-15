import axios from 'axios';
// Docs: https://axios-http.com/docs/api_intro

// Utility to call o4-mini-high model for search/summarization
// Set O4_MODEL_API_KEY and O4_MODEL_API_URL in .env.local
export async function queryO4Model(prompt) {
  const apiKey = process.env.O4_MODEL_API_KEY;
  const apiUrl = process.env.O4_MODEL_API_URL;
  if (!apiKey || !apiUrl) throw new Error('Missing O4 model API config');

  const res = await axios.post(apiUrl, {
    prompt,
    model: 'o4-mini-high',
    max_tokens: 256,
  }, {
    headers: {
      'Authorization': `Bearer ${apiKey}`,
      'Content-Type': 'application/json',
    },
  });

  return res.data.choices?.[0]?.text || '';
}
