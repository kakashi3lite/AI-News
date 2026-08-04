import { NextResponse } from 'next/server';
import { z } from 'zod';
import { getWatchlistWithStories } from '../../../lib/signal';
import { getWatchlistItems, seedWatchlist } from '../../../lib/watchlist';
import prisma from '../../../lib/db';

export const dynamic = 'force-dynamic';

const AddItemSchema = z.object({
  name: z.string().trim().min(2, 'Name must be at least 2 characters').max(60, 'Name too long'),
  aliases: z.array(z.string().trim().min(1).max(40)).max(20).default([]),
  keywords: z.array(z.string().trim().min(1).max(40)).max(40).default([]),
  category: z.string().trim().max(30).default('technology'),
});

// Watchlist: GET returns companies with their matched news streams;
// POST adds a new tracked company/entity (validated).
export async function GET() {
  try {
    await seedWatchlist(); // ensure defaults exist
    const items = await getWatchlistWithStories({ storiesPerItem: 6 });
    return NextResponse.json({ items });
  } catch (error) {
    console.error('[/api/watchlist] error:', error);
    return NextResponse.json({ error: error.message }, { status: 500 });
  }
}

export async function POST(req) {
  try {
    const body = await req.json();
    const parsed = AddItemSchema.safeParse(body);
    if (!parsed.success) {
      return NextResponse.json(
        { error: parsed.error.issues[0]?.message || 'Invalid input' },
        { status: 400 }
      );
    }
    const { name, aliases, keywords, category } = parsed.data;

    const item = await prisma.watchlistItem.upsert({
      where: { name },
      update: { aliases: JSON.stringify(aliases), keywords: JSON.stringify(keywords), category },
      create: { name, aliases: JSON.stringify(aliases), keywords: JSON.stringify(keywords), category },
    });

    return NextResponse.json(
      { item: { id: String(item.id), name: item.name, category: item.category } },
      { status: 201 }
    );
  } catch (error) {
    console.error('[/api/watchlist] error:', error);
    return NextResponse.json({ error: error.message }, { status: 500 });
  }
}
