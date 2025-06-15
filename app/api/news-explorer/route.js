import { NextResponse } from "next/server";
import { fetchAllNews } from "../../../lib/newsFetcher";
import { queryO4Model } from "../../../lib/o4ModelClient";

// POST /api/news-explorer
// Receives chat messages, returns a deep-dive news answer
export async function POST(req) {
  try {
    const { messages } = await req.json();
    const lastMsg = messages[messages.length - 1]?.content || "";
    if (!lastMsg) {
      return NextResponse.json({ reply: "Please enter a topic or question." }, { status: 400 });
    }
    // Fetch news related to the topic
    const news = await fetchAllNews({ query: lastMsg });
    if (!news.length) {
      return NextResponse.json({ reply: "No news found for this topic. Try something else." });
    }
    // Summarize top 2-3 articles for a deep-dive answer
    const context = news
      .slice(0, 3)
      .map((a, i) => `${i + 1}. ${a.title}: ${a.description || a.content}`)
      .join("\n");
    const prompt = `You are an expert news assistant. Summarize and explain the latest news on this topic:\n${lastMsg}\n\nHere are the most relevant articles:\n${context}\n\nGive a clear, concise, and insightful answer for a curious reader.`;
    const summary = await queryO4Model(prompt);
    return NextResponse.json({ reply: summary });
  } catch (e) {
    return NextResponse.json({ reply: "Sorry, an error occurred. Please try again." }, { status: 500 });
  }
}
