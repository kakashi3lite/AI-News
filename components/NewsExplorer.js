"use client";
import React, { useState, useRef, useEffect } from "react";
import { Popover, PopoverTrigger, PopoverContent } from "../components/ui/index";
import { Button } from "../components/ui/index";
import { Send } from "lucide-react";

// NewsExplorer: Floating chat popover for deep-dive news exploration
// Clean, modular, accessible, and documented
export default function NewsExplorer() {
  const [open, setOpen] = useState(false);
  const [messages, setMessages] = useState([
    { role: "system", content: "Ask me about any news topic to explore!" },
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const inputRef = useRef(null);
  const chatEndRef = useRef(null);

  // Scroll to last message
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, open]);

  // Handle sending a message
  async function handleSend(e) {
    e.preventDefault();
    if (!input.trim()) return;
    const userMsg = { role: "user", content: input };
    setMessages((msgs) => [...msgs, userMsg]);
    setInput("");
    setLoading(true);
    try {
      const res = await fetch("/api/news-explorer", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ messages: [...messages, userMsg] }),
      });
      const data = await res.json();
      setMessages((msgs) => [...msgs, { role: "assistant", content: data.reply }]);
    } catch (err) {
      setMessages((msgs) => [
        ...msgs,
        { role: "assistant", content: "Sorry, something went wrong. Please try again." },
      ]);
    }
    setLoading(false);
  }

  // Keyboard accessibility: Enter to send, Esc to close
  function handleKeyDown(e) {
    if (e.key === "Enter" && !e.shiftKey) {
      handleSend(e);
    } else if (e.key === "Escape") {
      setOpen(false);
    }
  }

  return (
    <div>
      {/* Floating Explore button */}
      <div
        style={{
          position: "fixed",
          bottom: 32,
          right: 32,
          zIndex: 50,
        }}
      >
        <Popover open={open} onOpenChange={setOpen}>
          <PopoverTrigger asChild>
            <Button
              variant="default"
              aria-label="Open News Explorer"
              style={{ borderRadius: "50%", width: 56, height: 56, boxShadow: "0 2px 8px #0002" }}
            >
              📰
            </Button>
          </PopoverTrigger>
          <PopoverContent
            align="end"
            sideOffset={12}
            style={{ width: 340, maxWidth: "90vw", padding: 0 }}
            className="shadow-lg rounded-lg bg-background border"
            aria-label="News Explorer Chat"
          >
            <div style={{ display: "flex", flexDirection: "column", height: 440 }}>
              {/* Chat messages */}
              <div
                style={{ flex: 1, overflowY: "auto", padding: 16 }}
                aria-live="polite"
                tabIndex={0}
              >
                {messages.map((msg, i) => (
                  <div
                    key={i}
                    style={{
                      marginBottom: 12,
                      textAlign: msg.role === "user" ? "right" : "left",
                    }}
                  >
                    <span
                      style={{
                        display: "inline-block",
                        padding: "8px 12px",
                        borderRadius: 16,
                        background:
                          msg.role === "user" ? "#2563eb22" : "#f3f4f6",
                        color: msg.role === "user" ? "#2563eb" : "#111",
                        fontSize: 15,
                        maxWidth: 240,
                        wordBreak: "break-word",
                      }}
                    >
                      {msg.content}
                    </span>
                  </div>
                ))}
                {loading && (
                  <div style={{ marginBottom: 12, textAlign: "left" }}>
                    <span style={{ fontStyle: "italic", color: "#888" }}>
                      Thinking...
                    </span>
                  </div>
                )}
                <div ref={chatEndRef} />
              </div>
              {/* Input box */}
              <form
                onSubmit={handleSend}
                style={{ display: "flex", borderTop: "1px solid #eee", padding: 8, background: "#fafbfc" }}
                autoComplete="off"
              >
                <input
                  ref={inputRef}
                  type="text"
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={handleKeyDown}
                  placeholder="Ask about any news topic..."
                  style={{
                    flex: 1,
                    border: "none",
                    outline: "none",
                    padding: 10,
                    fontSize: 15,
                    background: "transparent",
                  }}
                  aria-label="Type your message"
                  disabled={loading}
                />
                <button
                  type="submit"
                  aria-label="Send"
                  disabled={!input.trim() || loading}
                  style={{
                    background: "none",
                    border: "none",
                    cursor: loading ? "not-allowed" : "pointer",
                    padding: 6,
                  }}
                >
                  <Send size={22} color="#2563eb" />
                </button>
              </form>
            </div>
          </PopoverContent>
        </Popover>
      </div>
    </div>
  );
}

// ---
// This component provides a floating, accessible chat popover for deep news exploration.
// It follows modern dashboard UI/UX best practices (see README for details).
// Integrate <NewsExplorer /> at the root level (e.g., in app/page.js or layout.js).
