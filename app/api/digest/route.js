import { NextResponse } from 'next/server';
import { getDigest } from '../../../lib/signal';
import { ensureData } from '../../../lib/ingest';

// Daily digest: today's top themes, top stories, and watchlist pulse.
export async function GET() {
  try {
    await ensureData();
    const digest = await getDigest();
    return NextResponse.json(digest);
  } catch (error) {
    console.error('[/api/digest] error:', error);
    return NextResponse.json({ error: error.message }, { status: 500 });
  }
}
