import { NextResponse } from 'next/server';
import { fetchYouTubeTranscript } from '../../../lib/youtubeTranscript';
import { queryO4Model } from '../../../lib/o4ModelClient';
import { queryOpenAI } from '../../../lib/openaiClient';
import axios from 'axios';
import { spawn } from 'node:child_process';
import FormData from 'form-data';

/**
 * POST /api/summarize-youtube
 * Body: { url, engine }
 * Returns: { summary }
 */
/**
 * Transcribe audio via OpenAI Whisper
 */
async function transcribeAudioFromYouTube(videoId) {
  const url = `https://www.youtube.com/watch?v=${videoId}`;
  // Download audio using yt-dlp CLI
  const proc = spawn('yt-dlp', ['-f', 'bestaudio', '-o', '-', url]);
  const form = new FormData();
  form.append('file', proc.stdout, { filename: 'audio.mp3' });
  form.append('model', 'whisper-1');
  const res = await axios.post('https://api.openai.com/v1/audio/transcriptions', form, {
    headers: {
      ...form.getHeaders(),
      Authorization: `Bearer ${process.env.OPENAI_API_KEY}`
    },
    maxContentLength: Number.POSITIVE_INFINITY,
    maxBodyLength: Number.POSITIVE_INFINITY,
  });
  return res.data.text;
}

export async function POST(req) {
  try {
    const { url, engine } = await req.json();
    const modelEng = engine || 'o4';
    if (!url) return NextResponse.json({ summary: '' });
    // Extract video ID from URL (support watch?v=, embed/, youtu.be/)
    const regex = /(?:youtube\.com\/(?:watch\?(?:.*&)?v=|embed\/)|youtu\.be\/)([A-Za-z0-9_-]{11})/;
    const match = url.match(regex);
    const videoId = match ? match[1] : null;
    if (!videoId) {
      console.error(`[summarize-youtube] Invalid URL provided: ${url}`);
      return NextResponse.json({ summary: 'Invalid YouTube URL.' });
    }
    // Fetch transcript (captions first, then audio if openai engine)
    let transcript = await fetchYouTubeTranscript(videoId);
    if (!transcript && modelEng === 'openai') {
      try {
        transcript = await transcribeAudioFromYouTube(videoId);
      } catch (e) {
        console.error('[summarize-youtube] audio transcription error', e);
      }
    }
    if (!transcript) return NextResponse.json({ summary: 'Transcript not available.' });
    const promptText = `Summarize this YouTube video transcript into a concise news bite:\n${transcript}`;
    const summary = modelEng === 'openai'
      ? await queryOpenAI(promptText)
      : await queryO4Model(promptText);
    return NextResponse.json({ summary });
  } catch (error) {
    return NextResponse.json({ summary: 'Error summarizing YouTube video.' }, { status: 500 });
  }
}
