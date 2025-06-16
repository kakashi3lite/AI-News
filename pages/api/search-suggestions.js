// API endpoint for AI-powered search suggestions
import { openaiClient } from '../../lib/openaiClient';

export default async function handler(req, res) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  try {
    const { query } = req.body;

    if (!query || query.trim().length === 0) {
      return res.status(400).json({ error: 'Query is required' });
    }

    // Fallback suggestions for when AI is unavailable
    const fallbackSuggestions = [
      `${query} news`,
      `${query} analysis`,
      `${query} trends`,
      `${query} updates`,
      `${query} insights`
    ].slice(0, 5);

    try {
      // Try to get AI-powered suggestions
      const completion = await openaiClient.chat.completions.create({
        model: 'gpt-3.5-turbo',
        messages: [
          {
            role: 'system',
            content: 'You are a helpful assistant that generates relevant search suggestions for a news dashboard. Provide 5 concise, relevant search suggestions based on the user query. Return only the suggestions, one per line, without numbering or bullet points.'
          },
          {
            role: 'user',
            content: `Generate 5 search suggestions for: "${query}"`
          }
        ],
        max_tokens: 150,
        temperature: 0.7
      });

      const aiSuggestions = completion.choices[0]?.message?.content
        ?.split('\n')
        .filter(s => s.trim().length > 0)
        .slice(0, 5) || fallbackSuggestions;

      return res.status(200).json({
        suggestions: aiSuggestions,
        source: 'ai'
      });
    } catch (aiError) {
      console.warn('AI suggestions failed, using fallback:', aiError.message);
      
      // Return fallback suggestions if AI fails
      return res.status(200).json({
        suggestions: fallbackSuggestions,
        source: 'fallback'
      });
    }
  } catch (error) {
    console.error('Search suggestions error:', error);
    return res.status(500).json({ error: 'Internal server error' });
  }
}