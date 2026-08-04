import { NextResponse } from 'next/server';
import { getThemesWithStories } from '../../../lib/signal';
import { extractThemesFromDb } from '../../../lib/themes';
import prisma from '../../../lib/db';

// Trending theme clusters with representative stories.
// Re-extracts when stale (>1h) so the dashboard reflects current data.
export async function GET(req) {
  try {
    const force = new URL(req.url).searchParams.get('refresh') === '1';
    const newest = await prisma.theme.findFirst({ orderBy: { updatedAt: 'desc' } });
    const stale = force || !newest || Date.now() - new Date(newest.updatedAt).getTime() > 3600000;

    if (stale) {
      await extractThemesFromDb({ windowHours: 24 });
    }

    const themes = await getThemesWithStories({ limit: 12, storiesPerTheme: 5 });
    return NextResponse.json({ themes, refreshed: stale });
  } catch (error) {
    console.error('[/api/themes] error:', error);
    return NextResponse.json({ error: error.message }, { status: 500 });
  }
}
