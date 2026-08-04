/**
 * Client-side local user vault — works fully on static hosting (GitHub Pages).
 *
 * This is a LOCAL profile (device-scoped), not server authentication:
 * it keeps a user's research — bookmarks, personal watchlist, reading history —
 * and powers personalized recommendations. PINs are hashed with WebCrypto
 * (SHA-256 + salt) so plaintext is never stored.
 */

const USERS_KEY = 'ms_users_v1'; // { [email]: { salt, pinHash, profile } }
const SESSION_KEY = 'ms_session_v1'; // current email
const DATA_KEY = (email) => `ms_data_v1_${email}`; // { bookmarks, watchlist, history }

function readJSON(key, fallback) {
  try {
    const raw = localStorage.getItem(key);
    return raw ? JSON.parse(raw) : fallback;
  } catch {
    return fallback;
  }
}

function writeJSON(key, value) {
  try {
    localStorage.setItem(key, JSON.stringify(value));
  } catch {
    // storage full / unavailable — ignore
  }
}

export async function hashPin(pin, salt) {
  const data = new TextEncoder().encode(`${salt}:${pin}`);
  const buf = await crypto.subtle.digest('SHA-256', data);
  return [...new Uint8Array(buf)].map((b) => b.toString(16).padStart(2, '0')).join('');
}

function getUsers() {
  return readJSON(USERS_KEY, {});
}

export function getSession() {
  const email = localStorage.getItem(SESSION_KEY);
  if (!email) return null;
  const rec = getUsers()[email];
  return rec
    ? { email, name: rec.profile.name, createdAt: rec.profile.createdAt }
    : null;
}

export async function signUp({ email, name, pin }) {
  const e = String(email || '').trim().toLowerCase();
  const n = String(name || '').trim();
  if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(e)) return { error: 'Enter a valid email address.' };
  if (n.length < 2 || n.length > 40) return { error: 'Name must be 2–40 characters.' };
  if (!/^\d{4,8}$/.test(String(pin || ''))) return { error: 'PIN must be 4–8 digits.' };

  const users = getUsers();
  if (users[e]) return { error: 'That email is already registered. Sign in instead.' };

  const salt = `${Math.random().toString(36).slice(2)}${Date.now().toString(36)}`;
  const pinHash = await hashPin(String(pin), salt);
  users[e] = { salt, pinHash, profile: { name: n, email: e, createdAt: new Date().toISOString() } };
  writeJSON(USERS_KEY, users);
  localStorage.setItem(SESSION_KEY, e);
  return { ok: true, user: { email: e, name: n, createdAt: users[e].profile.createdAt } };
}

export async function logIn({ email, pin }) {
  const e = String(email || '').trim().toLowerCase();
  const rec = getUsers()[e];
  if (!rec) return { error: 'No account found with that email. Create one first.' };
  const pinHash = await hashPin(String(pin || ''), rec.salt);
  if (pinHash !== rec.pinHash) return { error: 'Incorrect PIN. Try again.' };
  localStorage.setItem(SESSION_KEY, e);
  return { ok: true, user: { email: e, name: rec.profile.name, createdAt: rec.profile.createdAt } };
}

export function logOut() {
  localStorage.removeItem(SESSION_KEY);
}

// ---------- per-user research data ----------

const emptyData = () => ({ bookmarks: [], watchlist: [], history: [], interests: {} });

export function loadUserData(email) {
  if (!email) return emptyData();
  const d = readJSON(DATA_KEY(email), emptyData());
  return {
    bookmarks: Array.isArray(d.bookmarks) ? d.bookmarks : [],
    watchlist: Array.isArray(d.watchlist) ? d.watchlist : [],
    history: Array.isArray(d.history) ? d.history : [],
    interests: d.interests && typeof d.interests === 'object' ? d.interests : {},
    crossword:
      d.crossword && typeof d.crossword === 'object'
        ? d.crossword
        : { lastSolved: null, streak: 0, solvedDates: [] },
  };
}

export function addBookmark(email, article) {
  const d = loadUserData(email);
  if (d.bookmarks.some((b) => b.id === article.id)) return d;
  d.bookmarks.unshift({
    id: article.id,
    title: article.title,
    url: article.url,
    tags: article.tags || [],
    category: article.category || 'general',
    source: article.source?.name || '',
    publishedAt: article.publishedAt || null,
    sentimentLabel: article.sentimentLabel || null,
  });
  d.bookmarks = d.bookmarks.slice(0, 60);
  writeJSON(DATA_KEY(email), d);
  return d;
}

export function removeBookmark(email, id) {
  const d = loadUserData(email);
  d.bookmarks = d.bookmarks.filter((b) => b.id !== id);
  writeJSON(DATA_KEY(email), d);
  return d;
}

export function addReading(email, article) {
  const d = loadUserData(email);
  d.history = d.history.filter((h) => h.id !== article.id);
  d.history.unshift({ id: article.id, title: article.title, at: Date.now() });
  d.history = d.history.slice(0, 100);
  writeJSON(DATA_KEY(email), d);
  return d;
}

export function addWatchlistItem(email, item) {
  const d = loadUserData(email);
  const name = String(item.name || '').trim();
  if (!name) return d;
  if (!d.watchlist.some((w) => w.name.toLowerCase() === name.toLowerCase())) {
    d.watchlist.push({
      name,
      keywords: Array.isArray(item.keywords) ? item.keywords : [],
      aliases: Array.isArray(item.aliases) ? item.aliases : [],
    });
    writeJSON(DATA_KEY(email), d);
  }
  return d;
}

export function removeWatchlistItem(email, name) {
  const d = loadUserData(email);
  d.watchlist = d.watchlist.filter((w) => w.name.toLowerCase() !== String(name).toLowerCase());
  writeJSON(DATA_KEY(email), d);
  return d;
}

/**
 * Auto-mode: learn interest weights from user activity.
 * Weights decay over time so the profile follows the user's CURRENT interests.
 */
export function addInterest(email, terms) {
  const d = loadUserData(email);
  const now = Date.now();
  d.interests = d.interests || {};
  // decay old interests
  for (const k of Object.keys(d.interests)) {
    d.interests[k] *= 0.93;
    if (d.interests[k] < 0.25) delete d.interests[k];
  }
  for (const t of terms || []) {
    const key = String(t).toLowerCase().trim();
    if (key.length < 2) continue;
    d.interests[key] = (d.interests[key] || 0) + 1;
  }
  void now;
  writeJSON(DATA_KEY(email), d);
  return d;
}

/** Top interest terms (weight desc). */
export function topInterests(email, limit = 8) {
  const d = loadUserData(email);
  return Object.entries(d.interests || {})
    .sort((a, b) => b[1] - a[1])
    .slice(0, limit)
    .map(([term, weight]) => ({ term, weight: Math.round(weight * 10) / 10 }));
}

// ---------- crossword habit (streak) ----------

/** Pure streak math — testable, no storage. Uses UTC day arithmetic. */
export function nextStreak(lastSolved, streak, today) {
  if (lastSolved === today) return { streak: streak || 0, sameDay: true };
  const [y, m, d] = today.split('-').map(Number);
  const yesterday = new Date(Date.UTC(y, m - 1, d) - 86400000).toISOString().slice(0, 10);
  if (lastSolved === yesterday) return { streak: (streak || 0) + 1, sameDay: false };
  return { streak: 1, sameDay: false };
}

export function todayStr() {
  return new Date().toISOString().slice(0, 10);
}

export function recordCrosswordSolved(email, date) {
  const d = loadUserData(email);
  const today = date || todayStr();
  d.crossword = d.crossword || { lastSolved: null, streak: 0, solvedDates: [] };
  const { streak, sameDay } = nextStreak(d.crossword.lastSolved, d.crossword.streak, today);
  if (!sameDay) {
    d.crossword.streak = streak;
    d.crossword.lastSolved = today;
    d.crossword.solvedDates = [...(d.crossword.solvedDates || []).slice(-60), today];
    writeJSON(DATA_KEY(email), d);
  }
  return d;
}

export function crosswordStatus(email) {
  const d = loadUserData(email);
  const today = todayStr();
  return {
    streak: d.crossword?.streak || 0,
    solvedToday: d.crossword?.lastSolved === today,
    lastSolved: d.crossword?.lastSolved || null,
    totalSolved: d.crossword?.solvedDates?.length || 0,
  };
}
