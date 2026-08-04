import { NextResponse } from 'next/server';
import { getDigest } from '../../../../lib/signal';
import { sendDigestEmail } from '../../../../lib/email';

// Scheduled digest email (Vercel Cron: GET /api/cron/digest).
// Protected by a Bearer token (CRON_SECRET). Builds today's digest and emails
// DIGEST_EMAILS via SMTP. Skips cleanly when SMTP isn't configured.
export async function GET(req) {
  const auth = req.headers.get('authorization') || '';
  const secret = process.env.CRON_SECRET;
  if (!secret || auth !== `Bearer ${secret}`) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
  }

  try {
    const digest = await getDigest();
    const result = await sendDigestEmail(digest);
    return NextResponse.json({ ...result, generatedAt: digest.generatedAt });
  } catch (error) {
    console.error('[/api/cron/digest] error:', error);
    return NextResponse.json({ error: error.message }, { status: 500 });
  }
}
