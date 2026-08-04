'use client';

import { useState } from 'react';
import { X, Loader2, Lock, Mail, User as UserIcon } from 'lucide-react';
import { useUser } from '../../contexts/UserContext';

// Clean login / create-account modal. Local profile vault (device-scoped).
export default function LoginModal() {
  const { loginOpen, setLoginOpen, handleLogin } = useUser();
  const [mode, setMode] = useState('signin');
  const [email, setEmail] = useState('');
  const [name, setName] = useState('');
  const [pin, setPin] = useState('');
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);

  if (!loginOpen) return null;

  const submit = async (e) => {
    e.preventDefault();
    setError('');
    setBusy(true);
    const res = await handleLogin({ email, name, pin, mode });
    if (res && res.error) setError(res.error);
    setBusy(false);
  };

  const switchMode = (m) => {
    setMode(m);
    setError('');
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4"
      onClick={() => setLoginOpen(false)}
    >
      <div
        className="w-full max-w-sm rounded-2xl bg-white shadow-2xl p-6"
        onClick={(ev) => ev.stopPropagation()}
        role="dialog"
        aria-modal="true"
      >
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-bold text-gray-900">
            {mode === 'signup' ? 'Create your research profile' : 'Sign in'}
          </h2>
          <button
            onClick={() => setLoginOpen(false)}
            className="text-gray-400 hover:text-gray-600"
            aria-label="Close"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        <p className="text-sm text-gray-500 mb-4">
          Keep your saved stories, watchlist, and history — and get a personalized
          “For You” feed on every login. Stored privately on this device.
        </p>

        {/* Mode toggle */}
        <div className="grid grid-cols-2 gap-1 rounded-lg bg-gray-100 p-1 mb-4">
          <button
            onClick={() => switchMode('signin')}
            className={`rounded-md py-1.5 text-sm font-medium ${
              mode === 'signin' ? 'bg-white text-blue-700 shadow-sm' : 'text-gray-500'
            }`}
          >
            Sign in
          </button>
          <button
            onClick={() => switchMode('signup')}
            className={`rounded-md py-1.5 text-sm font-medium ${
              mode === 'signup' ? 'bg-white text-blue-700 shadow-sm' : 'text-gray-500'
            }`}
          >
            Create account
          </button>
        </div>

        <form onSubmit={submit} className="space-y-3">
          {mode === 'signup' && (
            <div className="relative">
              <UserIcon className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
              <input
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="Your name"
                autoComplete="name"
                className="w-full rounded-lg border border-gray-300 pl-9 pr-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
          )}
          <div className="relative">
            <Mail className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
            <input
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="Email"
              type="email"
              autoComplete="email"
              className="w-full rounded-lg border border-gray-300 pl-9 pr-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>
          <div className="relative">
            <Lock className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
            <input
              value={pin}
              onChange={(e) => setPin(e.target.value)}
              placeholder="PIN (4–8 digits)"
              type="password"
              inputMode="numeric"
              autoComplete="current-password"
              className="w-full rounded-lg border border-gray-300 pl-9 pr-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>

          {error && (
            <div className="rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700">
              {error}
            </div>
          )}

          <button
            type="submit"
            disabled={busy}
            className="w-full rounded-lg bg-blue-600 py-2 text-sm font-semibold text-white hover:bg-blue-700 disabled:opacity-60 inline-flex items-center justify-center gap-2"
          >
            {busy && <Loader2 className="w-4 h-4 animate-spin" />}
            {mode === 'signup' ? 'Create profile' : 'Sign in'}
          </button>
        </form>

        <p className="mt-4 text-[11px] text-gray-400 text-center">
          Local profile — sign-in is device-scoped (suitable for static hosting).
        </p>
      </div>
    </div>
  );
}
