'use client';

import { createContext, useContext, useMemo, useState, useEffect, useCallback } from 'react';
import * as store from '../lib/client/userStore';

const UserContext = createContext(null);

export function UserProvider({ children }) {
  const [user, setUser] = useState(null);
  const [data, setData] = useState({ bookmarks: [], watchlist: [], history: [] });
  const [loginOpen, setLoginOpen] = useState(false);

  // Restore session on mount (client-only — safe for static export).
  useEffect(() => {
    const s = store.getSession();
    if (s) {
      setUser(s);
      setData(store.loadUserData(s.email));
    }
  }, []);

  const applySession = useCallback((u) => {
    setUser(u);
    setData(u ? store.loadUserData(u.email) : { bookmarks: [], watchlist: [], history: [] });
  }, []);

  const handleLogin = useCallback(
    async ({ email, name, pin, mode }) => {
      const res =
        mode === 'signup'
          ? await store.signUp({ email, name, pin })
          : await store.logIn({ email, pin });
      if (res.ok) {
        applySession(res.user);
        setLoginOpen(false);
      }
      return res;
    },
    [applySession]
  );

  const handleLogout = useCallback(() => {
    store.logOut();
    applySession(null);
  }, [applySession]);

  const toggleBookmark = useCallback(
    (article) => {
      if (!user) {
        setLoginOpen(true);
        return false;
      }
      const isB = data.bookmarks.some((b) => b.id === article.id);
      const next = isB
        ? store.removeBookmark(user.email, article.id)
        : store.addBookmark(user.email, article);
      setData(next);
      if (!isB) {
        // Auto-mode: learn from the bookmarked story.
        store.addInterest(user.email, [...(article.tags || []), article.category, article.source?.name]);
        setData(store.loadUserData(user.email));
      }
      return true;
    },
    [user, data.bookmarks]
  );

  const logReading = useCallback(
    (article) => {
      if (user) {
        store.addReading(user.email, article);
        store.addInterest(user.email, [...(article.tags || []), article.category, article.source?.name]);
      }
    },
    [user]
  );

  const addToWatchlist = useCallback(
    (item) => {
      if (!user) {
        setLoginOpen(true);
        return false;
      }
      setData(store.addWatchlistItem(user.email, item));
      // Auto-mode: the watched company is a strong interest signal.
      store.addInterest(user.email, [item.name, ...(item.keywords || [])]);
      setData(store.loadUserData(user.email));
      return true;
    },
    [user]
  );

  const removeFromWatchlist = useCallback(
    (name) => {
      if (user) setData(store.removeWatchlistItem(user.email, name));
    },
    [user]
  );

  // Crossword habit: streaks work for guests too (device-scoped).
  const recordCrosswordSolved = useCallback(
    (date) => {
      const email = user?.email || 'guest';
      const next = store.recordCrosswordSolved(email, date);
      if (user) setData(store.loadUserData(user.email));
      return next;
    },
    [user]
  );

  const crosswordStatus = useCallback(
    () => store.crosswordStatus(user?.email || 'guest'),
    [user]
  );

  const value = useMemo(
    () => ({
      user,
      data,
      loginOpen,
      setLoginOpen,
      isBookmarked: (id) => data.bookmarks.some((b) => b.id === id),
      toggleBookmark,
      logReading,
      addToWatchlist,
      removeFromWatchlist,
      handleLogin,
      handleLogout,
      recordCrosswordSolved,
      crosswordStatus,
    }),
    [
      user,
      data,
      loginOpen,
      toggleBookmark,
      logReading,
      addToWatchlist,
      removeFromWatchlist,
      handleLogin,
      handleLogout,
      recordCrosswordSolved,
      crosswordStatus,
    ]
  );

  return <UserContext.Provider value={value}>{children}</UserContext.Provider>;
}

export function useUser() {
  const ctx = useContext(UserContext);
  if (!ctx) throw new Error('useUser must be used within UserProvider');
  return ctx;
}
