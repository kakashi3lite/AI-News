import { NextResponse } from 'next/server';
import { queryOpenAI } from '../../../lib/openaiClient';

// POST /api/summarize-openai
// Receives { content } and returns { summary } using OpenAI
export async function POST(req) {
  try {
    const { content } = await req.json();
    if (!content) return NextResponse.json({ summary: '' });
    const summary = await queryOpenAI(`Summarize this news article: ${content}`);
    return NextResponse.json({ summary });
  } catch (error) {
    console.error('[summarize-openai]', error);
    return NextResponse.json({ summary: 'Error summarizing with OpenAI.' }, { status: 500 });
  }
}
