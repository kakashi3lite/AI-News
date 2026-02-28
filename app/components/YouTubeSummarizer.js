"use client";

import React, { useState } from "react";
import { Input } from "../../components/ui/input";
import { Youtube, Loader2, AlertCircle } from "lucide-react";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "../../components/ui/dropdown-menu";
import { Button } from "../../components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "../../components/ui/card";
import { Popover, PopoverTrigger, PopoverContent } from "../../components/ui/popover";

const YouTubeSummarizer = ({ modelEngine, setModelEngine }) => {
  const [ytUrl, setYtUrl] = useState("");
  const [ytSummary, setYtSummary] = useState("");
  const [ytLoading, setYtLoading] = useState(false);
  const [ytError, setYtError] = useState("");
  const [ytPopoverOpen, setYtPopoverOpen] = useState(false);

  const handleYtSummarize = async (e) => {
    e.preventDefault();
    setYtSummary("");
    setYtError("");
    setYtLoading(true);
    setYtPopoverOpen(true);
    try {
      const res = await fetch("/api/summarize-youtube", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url: ytUrl, engine: modelEngine }),
      });
      const data = await res.json();
      setYtSummary(data.summary || "");
      if (!data.summary) setYtError("Could not summarize video.");
    } catch {
      setYtError("Failed to summarize YouTube video.");
    }
    setYtLoading(false);
  };

  return (
    <Card className="border-0 rounded-none shadow-none">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Youtube className="w-5 h-5 text-red-600" /> YouTube News Bite
        </CardTitle>
        <CardDescription>Paste a YouTube URL to get an AI-generated news summary.</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="flex justify-end mb-4">
          <Button variant="outline" size="sm" onClick={() => setYtUrl("https://www.youtube.com/watch?v=dQw4w9WgXcQ")}>
            Use Sample URL
          </Button>
        </div>
        <Popover open={ytPopoverOpen} onOpenChange={setYtPopoverOpen}>
          <form onSubmit={handleYtSummarize} className="flex flex-col sm:flex-row gap-2">
            <Input value={ytUrl} onChange={(e) => setYtUrl(e.target.value)} placeholder="YouTube URL" />
            <PopoverTrigger asChild>
              <Button type="submit">Summarize</Button>
            </PopoverTrigger>
          </form>
          <PopoverContent className="w-80 p-4">
            <div className="flex items-center gap-2 mb-2">
              <span className="text-sm font-medium">Engine:</span>
              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <Button variant="outline" size="xs">{modelEngine === "openai" ? "OpenAI" : "O4"}</Button>
                </DropdownMenuTrigger>
                <DropdownMenuContent>
                  <DropdownMenuItem onSelect={() => setModelEngine("o4")}>O4 Model</DropdownMenuItem>
                  <DropdownMenuItem onSelect={() => setModelEngine("openai")}>OpenAI</DropdownMenuItem>
                </DropdownMenuContent>
              </DropdownMenu>
            </div>
            <div className="max-h-40 overflow-auto">
              {ytLoading ? <Loader2 className="h-6 w-6 animate-spin" /> : ytError ? <span>{ytError}</span> : <div className="whitespace-pre-wrap text-sm">{ytSummary}</div>}
            </div>
          </PopoverContent>
        </Popover>
        {ytError && (
          <Card className="mt-4 p-4 bg-red-50 border-red-200">
            <div className="flex items-center text-red-700">
              <AlertCircle className="w-5 h-5 mr-2" />
              <span>{ytError}</span>
            </div>
          </Card>
        )}
      </CardContent>
    </Card>
  );
};

export default YouTubeSummarizer;
