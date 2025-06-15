import { YouTubeTranscript } from 'youtube-transcript';
import axios from 'axios';

/**
 * Fetches the transcript of a YouTube video using a public transcript API (or fallback).
 * @param {string} videoId
 * @returns {Promise<string>} transcript
 */
export async function fetchYouTubeTranscript(videoId) {
  // Primary: fetch via youtube-transcript library
  try {
    const list = await YouTubeTranscript.fetchTranscript(videoId);
    if (list?.length) {
      return list.map(seg => seg.text).join(' ');
    }
  } catch (e) {
    console.error(`[youtubeTranscript] Primary fetch error for ${videoId}:`, e);
  }
  // Fallback: use public API
  try {
    const url = `https://yt.lemnoslife.com/noKey/yt_transcript?video_id=${videoId}`;
    const res = await axios.get(url);
    if (res.data?.transcripts?.length) {
      return res.data.transcripts.map(seg => seg.text).join(' ');
    }
  } catch (e) {
    console.error(`[youtubeTranscript] Fallback error for ${videoId}:`, e);
  }
  return '';
}
