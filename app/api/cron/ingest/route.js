import { NextResponse } from 'next/server';
import { ingestAll } from '../../../../lib/ingest';
import { extractThemesFromDb } from '../../../../lib/themes';
import { relinkAllWatchlist } from '../../../../lib/watchlist';

// Scheduled ingestion (Vercel Cron: GET /api/cron/ingest).
// Runs the full pipeline then refreshes themes + watchlist links.
// Protect with a Bearer token via CRON_SECRET env var in production.
export async function GET(req) {
  const secret = process.env.CRON_SECRET;
  if (secret && req.headers.get('authorization') !== `Bearer ${secret}`) {
    return NextResponse.json({ error: 'unauthorized' }, { status: 401 });
  }

  try {
    const result = await ingestAll();
    if (result.stats) {
      try {
        await relinkAllWatchlist();
        await extractThemesFromDb({ windowHours: 24 });
      } catch (err) {
        console.warn('[/api/cron/ingest] post-process failed:', err.message);
      }
    }
    return NextResponse.json(result);
  } catch (error) {
    console.error('[/api/cron/ingest] error:', error);
    return NextResponse.json({ error: error.message }, { status: 500 });
  }
}
