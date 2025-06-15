import axios from 'axios';

// Fetch and normalize news from Google News API (via Custom Search)
export async function fetchAllNews({ query = '', category = '', tag = '' }) {
  const apiKey = process.env.NEXT_PUBLIC_NEWS_API_KEY;
  const cx = process.env.GOOGLE_CSE_ID || 'news'; // IMPORTANT: Replace 'news' with your actual CSE ID if available
  let articles = [];
  let error = null;

  console.log(`[newsFetcher] Fetching news with query: '${query}', category: '${category}', tag: '${tag}', cx: '${cx}'`);

  try {
    let searchQuery = query || '';
    if (category) searchQuery += ` ${category}`; // Simple concatenation for now
    if (tag) searchQuery += ` ${tag}`; // Simple concatenation for now
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
  }

  console.log(`[newsFetcher] Returning ${articles.length} articles. Error: ${error || 'None'}`);
  return { articles, error };
}
