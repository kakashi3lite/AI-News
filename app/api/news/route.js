import { NextResponse } from 'next/server';
import { fetchAllNews } from '../../../lib/newsFetcher';

// API route to aggregate news from global sources and support search, category, and tag
export async function GET(req) {
  const { searchParams } = new URL(req.url);
  const query = searchParams.get('q') || '';
  const category = searchParams.get('category') || ''; // e.g., 'technology', 'business'
  const tag = searchParams.get('tag') || '';

  console.log(`[/api/news] Received request - Query: '${query}', Category: '${category}', Tag: '${tag}'`);

  try {
    // Fetch news from all sources
    const { articles, error } = await fetchAllNews({ query, category, tag });
    if (error) {
      console.error('[/api/news] Error from fetchAllNews:', error);
      return NextResponse.json({ error }, { status: 500 });
    }
    console.log(`[/api/news] Sending response with ${articles.length} articles.`);
    return NextResponse.json({ articles, totalResults: articles.length });
  } catch (error) {
    console.error('[/api/news] Unexpected error:', error);
    return NextResponse.json({ error: error.message }, { status: 500 });
  }
}
