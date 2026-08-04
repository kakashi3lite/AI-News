"use client";
import { useState } from 'react';
import { Youtube, Loader2, AlertCircle, Wand2 } from 'lucide-react';
import { Input } from '../ui/input';
import { Button } from '../ui/button';
import { Popover, PopoverTrigger, PopoverContent } from '../ui/popover';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '../ui/dropdown-menu';

// Secondary utility: summarize a YouTube video (kept as a power tool).
export default function ToolsView() {
  const [ytUrl, setYtUrl] = useState('');
  const [ytSummary, setYtSummary] = useState('');
  const [ytLoading, setYtLoading] = useState(false);
  const [ytError, setYtError] = useState('');
  const [ytPopoverOpen, setYtPopoverOpen] = useState(false);
  const [modelEngine, setModelEngine] = useState('o4');

  const handleYtSummarize = async (e) => {
    e.preventDefault();
    setYtSummary('');
    setYtError('');
    setYtLoading(true);
    setYtPopoverOpen(true);
    try {
      const res = await fetch('/api/summarize-youtube', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url: ytUrl, engine: modelEngine }),
      });
      const data = await res.json();
      setYtSummary(data.summary || '');
      if (!data.summary) setYtError('Could not summarize video.');
    } catch {
      setYtError('Failed to summarize YouTube video.');
    }
    setYtLoading(false);
  };

  return (
    <div className="space-y-6 max-w-3xl">
      <div>
        <h2 className="text-xl font-bold text-gray-900 flex items-center gap-2">
          <Wand2 className="w-5 h-5 text-blue-600" /> Tools
        </h2>
        <p className="text-sm text-gray-500 mt-0.5">
          Utility tools that support your market research.
        </p>
      </div>

      <div className="rounded-xl border border-gray-200 bg-white p-5">
        <h3 className="font-semibold text-gray-900 flex items-center gap-2 mb-1">
          <Youtube className="w-5 h-5 text-red-600" /> YouTube News Bite
        </h3>
        <p className="text-sm text-gray-500 mb-4">
          Paste a YouTube URL (e.g. an earnings call or product launch) to get an AI-generated
          summary.
        </p>

        <Popover open={ytPopoverOpen} onOpenChange={setYtPopoverOpen}>
          <form onSubmit={handleYtSummarize} className="flex flex-col sm:flex-row gap-2">
            <Input
              value={ytUrl}
              onChange={(e) => setYtUrl(e.target.value)}
              placeholder="https://www.youtube.com/watch?v=…"
              className="flex-1"
            />
            <PopoverTrigger asChild>
              <Button type="submit">Summarize</Button>
            </PopoverTrigger>
          </form>
          <PopoverContent className="w-96 p-4">
            <div className="flex items-center gap-2 mb-2">
              <span className="text-sm font-medium">Engine:</span>
              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <Button variant="outline" size="sm">
                    {modelEngine === 'openai' ? 'OpenAI' : 'O4'}
                  </Button>
                </DropdownMenuTrigger>
                <DropdownMenuContent>
                  <DropdownMenuItem onSelect={() => setModelEngine('o4')}>O4 Model</DropdownMenuItem>
                  <DropdownMenuItem onSelect={() => setModelEngine('openai')}>OpenAI</DropdownMenuItem>
                </DropdownMenuContent>
              </DropdownMenu>
            </div>
            <div className="max-h-48 overflow-auto">
              {ytLoading ? (
                <div className="flex items-center text-gray-500">
                  <Loader2 className="h-5 w-5 animate-spin mr-2" /> Summarizing…
                </div>
              ) : ytError ? (
                <div className="flex items-center text-red-600 text-sm">
                  <AlertCircle className="w-4 h-4 mr-1.5" /> {ytError}
                </div>
              ) : (
                <div className="whitespace-pre-wrap text-sm text-gray-700">{ytSummary}</div>
              )}
            </div>
          </PopoverContent>
        </Popover>
      </div>
    </div>
  );
}
