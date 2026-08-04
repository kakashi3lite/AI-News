import { NextResponse } from 'next/server';
import { ingestAll, isIngesting } from '../../../lib/ingest';
import { extractThemesFromDb } from '../../../lib/themes';
import { relinkAllWatchlist } from '../../../lib/watchlist';

// Simple in-memory rate limit: one manual refresh per 20s.
const RATE_LIMIT_MS = 20_000;
let lastIngestAt = 0;

// Manual ingestion trigger. Runs the full pipeline (fetch → dedup → persist
// → watchlist link → summarize), relinks watchlist for accuracy, and refreshes themes.
export async function POST() {
  const now = Date.now();
  if (!isIngesting() && now - lastIngestAt < RATE_LIMIT_MS) {
    return NextResponse.json(
      { error: 'Too many refresh requests. Please wait a few seconds.' },
      { status: 429 }
    );
  }
  try {
    lastIngestAt = now;
    const already = isIngesting();
    const result = await ingestAll();
    if (result.stats) {
      try {
        await relinkAllWatchlist();
        await extractThemesFromDb({ windowHours: 24 });
      } catch (err) {
        console.warn('[/api/ingest] post-process failed:', err.message);
      }
    }
    return NextResponse.json({ ...result, already });
  } catch (error) {
    console.error('[/api/ingest] error:', error);
    return NextResponse.json({ error: error.message }, { status: 500 });
  }
}

export async function GET() {
  return NextResponse.json({ running: isIngesting() });
}
