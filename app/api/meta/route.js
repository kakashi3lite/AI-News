import { NextResponse } from 'next/server';
import prisma from '../../../lib/db';

// Lightweight platform stats for the trust strip.
export async function GET() {
  try {
    const [sources, articles, verified] = await Promise.all([
      prisma.source.count({ where: { isActive: true } }),
      prisma.article.count(),
      prisma.article.count({
        where: { publishedAt: { gte: new Date(Date.now() - 24 * 3600000) } },
      }),
    ]);
    return NextResponse.json({
      sources,
      articles,
      last24h: verified,
      generatedAt: new Date().toISOString(),
    });
  } catch (error) {
    console.error('[/api/meta] error:', error);
    return NextResponse.json({ error: error.message }, { status: 500 });
  }
}
