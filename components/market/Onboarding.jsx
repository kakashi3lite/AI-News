"use client";

import { useEffect, useState } from 'react';
import { ShieldCheck, Sparkles, Grid3x3, X } from 'lucide-react';
import { useUser } from '../../contexts/UserContext';

const FLAG = 'ms_onboarded_v1';

const SLIDES = [
  {
    icon: ShieldCheck,
    title: 'Accuracy you can trust',
    body: 'Every story is source-graded, sentiment-tagged, verified across independent outlets, and scored for global market impact.',
  },
  {
    icon: Sparkles,
    title: 'A feed that learns you',
    body: 'Save stories, track competitors, and get a personal “For You” briefing that follows your interests automatically.',
  },
  {
    icon: Grid3x3,
    title: 'Learn today’s news with a crossword',
    body: 'A free crossword, rebuilt every day from real headlines. Solve it, keep your streak, stay sharp.',
  },
];

// First-run activation: three quick value slides (shown once, dismissible).
export default function Onboarding() {
  const { user, setLoginOpen } = useUser();
  const [open, setOpen] = useState(false);
  const [step, setStep] = useState(0);

  useEffect(() => {
    if (localStorage.getItem(FLAG)) return;
    const t = setTimeout(() => setOpen(true), 700);
    return () => clearTimeout(t);
  }, []);

  const done = () => {
    localStorage.setItem(FLAG, '1');
    setOpen(false);
  };

  if (!open) return null;
  const s = SLIDES[step];
  const Icon = s.icon;

  return (
    <div className="fixed inset-0 z-[60] flex items-center justify-center bg-black/50 p-4" onClick={done}>
      <div
        className="relative w-full max-w-md rounded-2xl bg-white p-7 shadow-2xl"
        onClick={(e) => e.stopPropagation()}
        role="dialog"
        aria-modal="true"
      >
        <button
          onClick={done}
          className="absolute top-4 right-4 text-gray-400 hover:text-gray-600"
          aria-label="Close"
        >
          <X className="w-5 h-5" />
        </button>

        <Icon className="w-10 h-10 text-blue-600 mb-4" />
        <h2 className="text-xl font-bold text-gray-900">{s.title}</h2>
        <p className="text-sm text-gray-600 mt-2 leading-relaxed">{s.body}</p>

        <div className="flex gap-1.5 mt-6">
          {SLIDES.map((_, i) => (
            <div key={i} className={`h-1.5 rounded-full transition-all ${i === step ? 'w-6 bg-blue-600' : 'w-3 bg-gray-200'}`} />
          ))}
        </div>

        <div className="flex justify-between items-center mt-6">
          <button onClick={done} className="text-sm text-gray-400 hover:text-gray-600">
            Skip
          </button>
          {step < SLIDES.length - 1 ? (
            <button
              onClick={() => setStep(step + 1)}
              className="rounded-lg bg-blue-600 px-5 py-2 text-sm font-semibold text-white hover:bg-blue-700"
            >
              Next
            </button>
          ) : (
            <button
              onClick={() => {
                done();
                if (!user) setLoginOpen(true);
              }}
              className="rounded-lg bg-blue-600 px-5 py-2 text-sm font-semibold text-white hover:bg-blue-700"
            >
              Get started
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
