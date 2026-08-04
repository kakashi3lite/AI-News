import { NextResponse } from 'next/server';
import { getDailyCrossword } from '../../../lib/crossword';

// Daily news crossword. Deterministic per date (cached in-memory), so the same
// puzzle is served all day and a new one appears the next day.
const cache = new Map();

function todayStr() {
  return new Date().toISOString().slice(0, 10);
}

export async function GET(req) {
  try {
    const date = new URL(req.url).searchParams.get('date') || todayStr();
    if (cache.has(date)) return NextResponse.json(cache.get(date));

    const puzzle = await getDailyCrossword({ date });
    cache.set(date, puzzle);
    return NextResponse.json(puzzle);
  } catch (error) {
    console.error('[/api/crossword] error:', error);
    return NextResponse.json({ error: error.message }, { status: 500 });
  }
}
