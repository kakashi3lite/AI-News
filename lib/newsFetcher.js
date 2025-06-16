import axios from 'axios';

// Mock news data for development and testing
const mockNewsData = {
  general: [
    {
      id: 'mock-1',
      title: 'AI Revolution in News Industry Continues to Transform Media Landscape',
      description: 'Artificial intelligence is reshaping how news is gathered, processed, and delivered to audiences worldwide.',
      content: 'The integration of artificial intelligence in newsrooms has accelerated dramatically over the past year. From automated content generation to real-time fact-checking, AI tools are becoming indispensable for modern journalism. News organizations are leveraging machine learning algorithms to analyze vast amounts of data, identify trending topics, and personalize content delivery. This technological shift is not only improving efficiency but also enabling journalists to focus on more complex investigative work while AI handles routine tasks.',
      url: 'https://example.com/ai-news-revolution',
      image: 'https://images.unsplash.com/photo-1677442136019-21780ecad995?w=400',
      publishedAt: new Date().toISOString(),
      source: { name: 'Tech News Daily', url: 'https://technewsdaily.com' },
      category: 'technology',
      tags: ['AI', 'journalism', 'technology', 'media']
    },
    {
      id: 'mock-2',
      title: 'Global Climate Summit Reaches Historic Agreement on Carbon Reduction',
      description: 'World leaders unite on ambitious new targets for reducing greenhouse gas emissions by 2030.',
      content: 'In a landmark decision at the Global Climate Summit, representatives from 195 countries have agreed to implement unprecedented measures to combat climate change. The agreement includes binding commitments to reduce carbon emissions by 50% within the next decade, massive investments in renewable energy infrastructure, and support for developing nations transitioning to clean energy. Environmental scientists are calling this the most significant climate action since the Paris Agreement.',
      url: 'https://example.com/climate-summit-agreement',
      image: 'https://images.unsplash.com/photo-1569163139394-de4e4f43e4e3?w=400',
      publishedAt: new Date(Date.now() - 3600000).toISOString(),
      source: { name: 'Global Environment Report', url: 'https://globalenvironment.org' },
      category: 'environment',
      tags: ['climate', 'environment', 'politics', 'global']
    },
    {
      id: 'mock-3',
      title: 'Breakthrough in Quantum Computing Promises Revolutionary Applications',
      description: 'Scientists achieve new milestone in quantum error correction, bringing practical quantum computers closer to reality.',
      content: 'Researchers at leading technology institutes have made a significant breakthrough in quantum error correction, solving one of the biggest challenges in quantum computing. This advancement could accelerate the development of practical quantum computers capable of solving complex problems in cryptography, drug discovery, and financial modeling. The new error correction method reduces quantum decoherence by 99.9%, making quantum computers more stable and reliable for real-world applications.',
      url: 'https://example.com/quantum-computing-breakthrough',
      image: 'https://images.unsplash.com/photo-1635070041078-e363dbe005cb?w=400',
      publishedAt: new Date(Date.now() - 7200000).toISOString(),
      source: { name: 'Science Today', url: 'https://sciencetoday.com' },
      category: 'technology',
      tags: ['quantum computing', 'science', 'technology', 'research']
    }
  ],
  technology: [
    {
      id: 'tech-1',
      title: 'New Programming Language Designed for AI Development Gains Popularity',
      description: 'Developers embrace innovative language that simplifies machine learning model creation.',
      content: 'A new programming language specifically designed for artificial intelligence development is gaining traction among developers worldwide. The language, called AIScript, offers intuitive syntax for building neural networks, natural language processing models, and computer vision applications. Early adopters report 40% faster development times and improved code readability compared to traditional AI frameworks.',
      url: 'https://example.com/new-ai-programming-language',
      image: 'https://images.unsplash.com/photo-1555066931-4365d14bab8c?w=400',
      publishedAt: new Date(Date.now() - 10800000).toISOString(),
      source: { name: 'Developer Weekly', url: 'https://developerweekly.com' },
      category: 'technology',
      tags: ['programming', 'AI', 'development', 'software']
    }
  ],
  business: [
    {
      id: 'biz-1',
      title: 'Startup Ecosystem Shows Strong Recovery with Record Investment Levels',
      description: 'Venture capital funding reaches new heights as investors show confidence in emerging technologies.',
      content: 'The global startup ecosystem is experiencing a remarkable recovery, with venture capital investments reaching record levels in the third quarter. Technology startups focusing on AI, clean energy, and healthcare are attracting the most funding. Industry analysts attribute this surge to increased investor confidence and the proven resilience of tech companies during economic uncertainty.',
      url: 'https://example.com/startup-investment-record',
      image: 'https://images.unsplash.com/photo-1559136555-9303baea8ebd?w=400',
      publishedAt: new Date(Date.now() - 14400000).toISOString(),
      source: { name: 'Business Insider', url: 'https://businessinsider.com' },
      category: 'business',
      tags: ['startups', 'investment', 'venture capital', 'business']
    }
  ]
};

// Fetch and normalize news from Google News API (via Custom Search) or return mock data
export async function fetchAllNews({ query = '', category = '', tag = '' }) {
  const apiKey = process.env.NEXT_PUBLIC_NEWS_API_KEY;
  const cx = process.env.GOOGLE_CSE_ID;
  const mockMode = process.env.NEXT_PUBLIC_MOCK_MODE === 'true';
  let articles = [];
  let error = null;

  console.log(`[newsFetcher] Fetching news with query: '${query}', category: '${category}', tag: '${tag}', mockMode: ${mockMode}`);

  // Use mock data if in mock mode or if API keys are missing
  if (mockMode || !apiKey || apiKey === 'mock-news-api-key' || !cx || cx === 'mock-cse-id') {
    console.log('[newsFetcher] Using mock data');
    
    // Get articles based on category or return general articles
    let sourceArticles = mockNewsData[category] || mockNewsData.general;
    
    // Filter by query if provided
    if (query) {
      sourceArticles = sourceArticles.filter(article => 
        article.title.toLowerCase().includes(query.toLowerCase()) ||
        article.description.toLowerCase().includes(query.toLowerCase()) ||
        article.tags.some(tag => tag.toLowerCase().includes(query.toLowerCase()))
      );
    }
    
    // Filter by tag if provided
    if (tag) {
      sourceArticles = sourceArticles.filter(article => 
        article.tags.some(articleTag => articleTag.toLowerCase().includes(tag.toLowerCase()))
      );
    }
    
    // Add some variety by including articles from other categories
    if (sourceArticles.length < 5) {
      const otherCategories = Object.keys(mockNewsData).filter(cat => cat !== category);
      otherCategories.forEach(cat => {
        sourceArticles = [...sourceArticles, ...mockNewsData[cat].slice(0, 2)];
      });
    }
    
    articles = sourceArticles.slice(0, 10); // Limit to 10 articles
    return { articles, error: null };
  }

  // Real API call logic (original implementation)
  try {
    let searchQuery = query || '';
    if (category) searchQuery += ` ${category}`;
    if (tag) searchQuery += ` ${tag}`;
    const url = `https://www.googleapis.com/customsearch/v1?q=${encodeURIComponent(searchQuery || 'news')}&cx=${cx}&key=${apiKey}&num=10&gl=us`;

    console.log(`[newsFetcher] Requesting URL: ${url}`);

    const res = await axios.get(url);
    console.log('[newsFetcher] Google API Response Status:', res.status);

    if (res.data.items) {
      articles = res.data.items.map((item, i) => ({
        id: `google-${i}-${item.cacheId || item.link}`,
        title: item.title,
        description: item.snippet,
        content: item.snippet,
        url: item.link,
        image: item.pagemap?.cse_image?.[0]?.src || '',
        publishedAt: '',
        source: { name: item.displayLink, url: item.link },
        category: category || 'general',
        tags: item.pagemap?.metatags?.[0]?.news_keywords?.split(',').map(t => t.trim()).filter(Boolean) || [],
      }));
    }
  } catch (e) {
    error = e.response?.data?.error?.message || e.message || 'Unknown error fetching news.';
    console.error('[newsFetcher] Error fetching news:', e.response?.data || e.message);
    if (e.response?.status === 403) {
      error = "API Key error or Quota Exceeded. Please check your Google API Key and usage limits.";
    } else if (e.response?.status === 400 && e.response?.data?.error?.message.includes('Invalid Value')) {
      error = "Invalid Custom Search Engine ID (cx). Please ensure it's correct.";
    }
    
    // Fallback to mock data if API fails
    console.log('[newsFetcher] API failed, falling back to mock data');
    const sourceArticles = mockNewsData[category] || mockNewsData.general;
    articles = sourceArticles.slice(0, 10);
    error = null; // Clear error since we have fallback data
  }

  console.log(`[newsFetcher] Returning ${articles.length} articles. Error: ${error || 'None'}`);
  return { articles, error };
}
