import { NextResponse } from 'next/server';
import { getTopStories } from '../../../lib/signal';
import { ensureData } from '../../../lib/ingest';

// DB-backed news feed for the Market Signal dashboard.
// Auto-ingests real RSS data on first load so the dashboard is never empty.
export async function GET(req) {
  const { searchParams } = new URL(req.url);
  const query = searchParams.get('q') || '';
  const category = searchParams.get('category') || '';
  const tag = searchParams.get('tag') || '';
  const watchlistId = searchParams.get('watchlist') || '';
  const limit = Math.min(Number(searchParams.get('limit')) || 20, 60);

  try {
    await ensureData();
    const articles = await getTopStories({
      query,
      category,
      tag,
      watchlistId,
      limit,
      sinceHours: 72,
    });
    return NextResponse.json({
      articles,
      totalResults: articles.length,
      generatedAt: new Date().toISOString(),
    });
  } catch (error) {
    console.error('[/api/news] error:', error);
    return NextResponse.json({ error: error.message }, { status: 500 });
  }
}
