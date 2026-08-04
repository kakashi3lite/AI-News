"use client";

import React, { useState } from 'react';
import {
  Newspaper,
  Activity,
  Building2,
  LayoutList,
  Mail,
  Wand2,
  RefreshCw,
  Loader2,
  Sparkles,
  LogIn,
  LogOut,
  Grid3x3,
  Sunrise,
} from 'lucide-react';
import SignalView from '../components/market/SignalView';
import WatchlistView from '../components/market/WatchlistView';
import StoriesView from '../components/market/StoriesView';
import DigestView from '../components/market/DigestView';
import ToolsView from '../components/market/ToolsView';
import ForYouView from '../components/market/ForYouView';
import CrosswordView from '../components/market/CrosswordView';
import TodayView from '../components/market/TodayView';
import Onboarding from '../components/market/Onboarding';
import LoginModal from '../components/market/LoginModal';
import { UserProvider, useUser } from '../contexts/UserContext';
import { triggerIngest } from '../lib/clientData';

const NAV = [
  { id: 'today', label: 'Today', icon: Sunrise },
  { id: 'foryou', label: 'For You', icon: Sparkles },
  { id: 'signal', label: 'Signal', icon: Activity },
  { id: 'watchlist', label: 'Watchlist', icon: Building2 },
  { id: 'crossword', label: 'Crossword', icon: Grid3x3 },
  { id: 'stories', label: 'Stories', icon: LayoutList },
  { id: 'digest', label: 'Digest', icon: Mail },
  { id: 'tools', label: 'Tools', icon: Wand2 },
];

function Header({ activeView, setActiveView, ingesting, onIngest, lastResult }) {
  const { user, setLoginOpen, handleLogout } = useUser();

  return (
    <div className="bg-gradient-to-r from-blue-600 to-indigo-700 rounded-xl px-6 py-5 mb-6 shadow-lg">
      <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4">
        <div className="flex items-center gap-3">
          <Newspaper className="w-9 h-9 text-white shrink-0" />
          <div>
            <h1 className="text-2xl font-bold text-white leading-tight">Market Signal</h1>
            <p className="text-blue-100 text-sm">
              Competitive intelligence dashboard — real-time, source-graded, sentiment-tagged.
            </p>
          </div>
        </div>

        <div className="flex items-center gap-3">
          {user ? (
            <div className="flex items-center gap-2 bg-white/15 border border-white/25 rounded-lg px-3 py-1.5">
              <span className="flex h-7 w-7 items-center justify-center rounded-full bg-white text-sm font-bold text-blue-700">
                {user.name.charAt(0).toUpperCase()}
              </span>
              <span className="text-sm font-medium text-white max-w-[140px] truncate">
                {user.name}
              </span>
              <button
                onClick={handleLogout}
                className="text-blue-100 hover:text-white"
                title="Sign out"
                aria-label="Sign out"
              >
                <LogOut className="w-4 h-4" />
              </button>
            </div>
          ) : (
            <button
              onClick={() => setLoginOpen(true)}
              className="inline-flex items-center gap-2 rounded-lg bg-white text-blue-700 px-4 py-2 text-sm font-semibold hover:bg-blue-50"
            >
              <LogIn className="w-4 h-4" /> Sign in
            </button>
          )}
          <button
            onClick={onIngest}
            disabled={ingesting}
            className="inline-flex items-center gap-2 rounded-lg bg-white/15 border border-white/25 px-4 py-2 text-sm font-medium text-white hover:bg-white/25 disabled:opacity-60"
          >
            {ingesting ? <Loader2 className="w-4 h-4 animate-spin" /> : <RefreshCw className="w-4 h-4" />}
            {ingesting ? 'Updating…' : 'Refresh data'}
          </button>
        </div>
      </div>

      {/* Nav */}
      <div className="mt-4 flex flex-wrap gap-2">
        {NAV.map(({ id, label, icon: Icon }) => (
          <button
            key={id}
            onClick={() => setActiveView(id)}
            className={`inline-flex items-center gap-2 rounded-lg px-3.5 py-1.5 text-sm font-medium transition-colors ${
              activeView === id ? 'bg-white text-blue-700 shadow-sm' : 'text-white hover:bg-white/20'
            }`}
          >
            <Icon className="w-4 h-4" />
            {label}
          </button>
        ))}
      </div>

      {lastResult && !ingesting && !lastResult.error && (
        <div className="mt-3 text-xs text-blue-100">
          Last update: {lastResult.inserted ?? 0} new · {lastResult.duplicates ?? 0} duplicates
          · {lastResult.linked ?? 0} watchlist matches
          {typeof lastResult.processingTimeMs === 'number'
            ? ` · ${Math.round(lastResult.processingTimeMs / 1000)}s`
            : ''}
        </div>
      )}
      {lastResult && !ingesting && lastResult.error && (
        <div className="mt-3 text-xs text-red-200">{lastResult.error}</div>
      )}
    </div>
  );
}

// Market Signal — competitive intelligence dashboard.
// Works in two modes: server (API + DB) and static (pre-built JSON snapshots).
// Includes a local user profile for saved research + personalized recommendations.
const MarketDashboard = () => {
  const [activeView, setActiveView] = useState('today');
  const [ingesting, setIngesting] = useState(false);
  const [lastResult, setLastResult] = useState(null);

  const handleIngest = async () => {
    if (ingesting) return;
    setIngesting(true);
    try {
      const data = await triggerIngest();
      setLastResult(data.stats || data);
    } catch {
      setLastResult({ error: 'Refresh failed.' });
    } finally {
      setIngesting(false);
    }
  };

  return (
    <UserProvider>
      <main className="min-h-screen bg-gray-50 flex flex-col items-center px-4 py-6">
        <div className="w-full max-w-7xl">
          <Header
            activeView={activeView}
            setActiveView={setActiveView}
            ingesting={ingesting}
            onIngest={handleIngest}
            lastResult={lastResult}
          />

          <div className="mb-10">
            {activeView === 'today' && <TodayView onOpenCrossword={() => setActiveView('crossword')} />}
            {activeView === 'foryou' && <ForYouView />}
            {activeView === 'signal' && <SignalView />}
            {activeView === 'watchlist' && <WatchlistView />}
            {activeView === 'crossword' && <CrosswordView />}
            {activeView === 'stories' && <StoriesView />}
            {activeView === 'digest' && <DigestView />}
            {activeView === 'tools' && <ToolsView />}
          </div>
        </div>
        <LoginModal />
        <Onboarding />
      </main>
    </UserProvider>
  );
};

export default MarketDashboard;
