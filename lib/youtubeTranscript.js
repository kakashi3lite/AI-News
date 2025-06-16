import { YouTubeTranscript } from 'youtube-transcript';
import axios from 'axios';

// Mock transcripts for development/testing when APIs are unavailable
const MOCK_TRANSCRIPTS = [
  "Welcome to today's tech news update. In this video, we'll be discussing the latest developments in artificial intelligence and machine learning. Recent breakthroughs in natural language processing have shown remarkable improvements in understanding context and generating human-like responses. Companies are investing heavily in AI research, with particular focus on ethical AI development and responsible deployment. The implications for various industries are significant, from healthcare to finance to education. We'll explore how these technologies are being integrated into everyday applications and what this means for the future of work and society.",
  "Breaking news in the world of technology today. A major software company has announced a significant update to their platform, introducing new features that promise to revolutionize how users interact with digital content. The update includes enhanced security measures, improved user interface design, and advanced analytics capabilities. Industry experts are calling this a game-changer for the sector. We'll dive deep into what these changes mean for businesses and consumers alike, and how this might influence competitor strategies in the coming months.",
  "Today we're covering the latest developments in renewable energy technology. Solar panel efficiency has reached new heights with recent innovations in photovoltaic cell design. Wind energy projects are expanding globally, with several countries announcing ambitious targets for clean energy adoption. The economic impact of this transition is substantial, creating new job opportunities while challenging traditional energy sectors. We'll examine the policy implications, technological challenges, and environmental benefits of these renewable energy initiatives.",
  "In this episode, we explore the evolving landscape of digital privacy and cybersecurity. Recent data breaches have highlighted the importance of robust security measures and user privacy protection. New regulations are being implemented worldwide to address these concerns, affecting how companies collect, store, and use personal data. We'll discuss the balance between innovation and privacy, the role of encryption in protecting user information, and what individuals can do to safeguard their digital presence in an increasingly connected world."
];

/**
 * Fetches the transcript of a YouTube video using a public transcript API (or fallback).
 * @param {string} videoId
 * @returns {Promise<string>} transcript
 */
export async function fetchYouTubeTranscript(videoId) {
  // Check if we should use mock data
  const isMockMode = process.env.USE_MOCK_DATA === 'true';
  const isDevelopment = process.env.NODE_ENV === 'development';
  
  if (isMockMode) {
    console.log('🔧 Using mock YouTube transcript (mock mode enabled)');
    const randomIndex = Math.floor(Math.random() * MOCK_TRANSCRIPTS.length);
    return MOCK_TRANSCRIPTS[randomIndex];
  }
  
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
    const res = await axios.get(url, { timeout: 10000 });
    if (res.data?.transcripts?.length) {
      return res.data.transcripts.map(seg => seg.text).join(' ');
    }
  } catch (e) {
    console.error(`[youtubeTranscript] Fallback error for ${videoId}:`, e);
  }
  
  // Final fallback: use mock data if all APIs fail
  if (isDevelopment) {
    console.warn('⚠️ YouTube transcript APIs failed, using mock data for development');
    const randomIndex = Math.floor(Math.random() * MOCK_TRANSCRIPTS.length);
    return MOCK_TRANSCRIPTS[randomIndex];
  }
  
  return '';
}
