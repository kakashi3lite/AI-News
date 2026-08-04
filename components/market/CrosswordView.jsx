"use client";

import { useEffect, useMemo, useRef, useState } from 'react';
import { Loader2, AlertCircle, Check, Eye, Eraser, Sparkles } from 'lucide-react';
import { fetchCrossword } from '../../lib/clientData';

// Daily news crossword — solved in-browser, generated from today's real headlines.
export default function CrosswordView() {
  const [puzzle, setPuzzle] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [entries, setEntries] = useState({});
  const [selected, setSelected] = useState(null);
  const [wrong, setWrong] = useState({});
  const [revealed, setRevealed] = useState(false);
  const boardRef = useRef(null);

  useEffect(() => {
    (async () => {
      try {
        const p = await fetchCrossword();
        if (p.error) setError(p.error);
        else setPuzzle(p);
      } catch {
        setError('Failed to load the crossword.');
      } finally {
        setLoading(false);
      }
    })();
  }, []);

  const wordPaths = useMemo(() => {
    if (!puzzle) return {};
    const map = {};
    const push = (w) => {
      const cells = [];
      for (let i = 0; i < w.word.length; i++) {
        cells.push(w.dir === 'across' ? [w.row, w.col + i] : [w.row + i, w.col]);
      }
      map[`${w.number}:${w.dir}`] = cells;
    };
    (puzzle.across || []).forEach(push);
    (puzzle.down || []).forEach(push);
    return map;
  }, [puzzle]);

  const focusCell = (r, c) => {
    const k = `${r},${c}`;
    if (puzzle && puzzle.cells[k]) {
      setSelected([r, c]);
      document.getElementById(`xw-${r}-${c}`)?.focus();
    }
  };

  const pathFor = (pos, dir) => {
    if (!pos) return null;
    const [r, c] = pos;
    const across = (puzzle.across || []).find(
      (w) => w.row === r && c >= w.col && c < w.col + w.word.length
    );
    if (across && dir !== 'down') return wordPaths[`${across.number}:across`];
    const down = (puzzle.down || []).find(
      (w) => w.col === c && r >= w.row && r < w.row + w.word.length
    );
    if (down) return wordPaths[`${down.number}:down`];
    return null;
  };

  const moveAlong = (delta, dir) => {
    if (!selected) return;
    const path = pathFor(selected, dir) || pathFor(selected);
    if (!path) return;
    const idx = path.findIndex(([r, c]) => r === selected[0] && c === selected[1]);
    const next = path[(idx + delta + path.length) % path.length];
    focusCell(next[0], next[1]);
  };

  const handleKeyDown = (e) => {
    if (!selected) return;
    const [r, c] = selected;
    const k = `${r},${c}`;

    if (/^[a-zA-Z]$/.test(e.key)) {
      e.preventDefault();
      setEntries((prev) => ({ ...prev, [k]: e.key.toUpperCase() }));
      setWrong((prev) => {
        const copy = { ...prev };
        delete copy[k];
        return copy;
      });
      moveAlong(1, 'across');
    } else if (e.key === 'Backspace') {
      e.preventDefault();
      if (entries[k]) {
        setEntries((prev) => {
          const copy = { ...prev };
          delete copy[k];
          return copy;
        });
      } else {
        moveAlong(-1, 'across');
      }
    } else if (e.key.startsWith('Arrow')) {
      e.preventDefault();
      const dir = e.key.replace('Arrow', '').toLowerCase();
      const [dr, dc] =
        dir === 'up' ? [-1, 0] : dir === 'down' ? [1, 0] : dir === 'left' ? [0, -1] : [0, 1];
      focusCell(r + dr, c + dc);
    }
  };

  const checkAnswers = () => {
    if (!puzzle) return;
    const bad = {};
    for (const [k, answer] of Object.entries(puzzle.cells)) {
      if ((entries[k] || '').toUpperCase() !== answer) bad[k] = true;
    }
    setWrong(bad);
    setRevealed(false);
  };

  const reveal = () => {
    if (!puzzle) return;
    const filled = {};
    for (const [k, answer] of Object.entries(puzzle.cells)) filled[k] = answer;
    setEntries(filled);
    setWrong({});
    setRevealed(true);
  };

  const clear = () => {
    setEntries({});
    setWrong({});
    setRevealed(false);
  };

  const solvedCount = puzzle
    ? Object.keys(puzzle.cells).filter((k) => (entries[k] || '').toUpperCase() === puzzle.cells[k]).length
    : 0;
  const totalCells = puzzle ? Object.keys(puzzle.cells).length : 0;

  if (loading) {
    return (
      <div className="flex justify-center items-center p-16 text-gray-500">
        <Loader2 className="h-6 w-6 animate-spin mr-2" /> Generating today&apos;s puzzle from the news…
      </div>
    );
  }

  if (error) {
    return (
      <div className="rounded-xl border border-red-200 bg-red-50 p-4 flex items-center text-red-700">
        <AlertCircle className="w-5 h-5 mr-2 shrink-0" />
        <span>{error}</span>
      </div>
    );
  }

  const size = puzzle.size;
  const clueLists = [
    { title: 'Across', items: puzzle.across },
    { title: 'Down', items: puzzle.down },
  ];

  return (
    <div className="space-y-6">
      <div className="flex items-start justify-between gap-4 flex-wrap">
        <div>
          <h2 className="text-xl font-bold text-gray-900 flex items-center gap-2">
            <Sparkles className="w-5 h-5 text-blue-600" /> Daily News Crossword
          </h2>
          <p className="text-sm text-gray-500 mt-0.5">
            Free, updated daily, built from {puzzle.generatedFrom}. Read today&apos;s news while you
            solve.
          </p>
        </div>
        <div className="text-sm text-gray-500 bg-white border border-gray-200 rounded-lg px-3 py-1.5">
          {solvedCount}/{totalCells} solved
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Board */}
        <div>
          <div
            ref={boardRef}
            className="grid gap-[3px] bg-gray-200 border border-gray-200 rounded-lg p-[3px] w-full max-w-[520px] mx-auto"
            style={{ gridTemplateColumns: `repeat(${size}, minmax(0, 1fr))` }}
          >
            {Array.from({ length: size }).map((_, r) =>
              Array.from({ length: size }).map((__, c) => {
                const k = `${r},${c}`;
                const filled = Boolean(puzzle.cells[k]);
                if (!filled) {
                  return <div key={k} className="bg-gray-50 aspect-square" />;
                }
                const num = puzzle.numbers[k];
                const isSelected = selected && selected[0] === r && selected[1] === c;
                const isWrong = wrong[k];
                return (
                  <div
                    key={k}
                    className={`relative aspect-square ${isWrong ? 'bg-red-100' : 'bg-white'}`}
                  >
                    {num && (
                      <span className="absolute top-0 left-0.5 text-[8px] leading-none text-gray-500 z-10">
                        {num}
                      </span>
                    )}
                    <input
                      id={`xw-${r}-${c}`}
                      value={entries[k] || ''}
                      onChange={(e) => {
                        const v = e.target.value.replace(/[^a-zA-Z]/g, '').slice(-1);
                        if (v) setEntries((prev) => ({ ...prev, [k]: v.toUpperCase() }));
                      }}
                      onKeyDown={handleKeyDown}
                      onFocus={() => setSelected([r, c])}
                      onClick={() => setSelected([r, c])}
                      aria-label={`Cell ${r + 1},${c + 1}`}
                      className={`w-full h-full text-center text-base sm:text-lg font-semibold uppercase text-gray-800 outline-none focus:bg-blue-100 focus:ring-2 focus:ring-blue-400 ${
                        isSelected ? 'bg-blue-50' : ''
                      }`}
                    />
                  </div>
                );
              })
            )}
          </div>

          <div className="flex flex-wrap gap-2 mt-4 justify-center">
            <button
              onClick={checkAnswers}
              className="inline-flex items-center gap-2 rounded-lg bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-700"
            >
              <Check className="w-4 h-4" /> Check
            </button>
            <button
              onClick={reveal}
              className="inline-flex items-center gap-2 rounded-lg border border-gray-300 bg-white px-4 py-2 text-sm font-medium text-gray-700 hover:bg-gray-50"
            >
              <Eye className="w-4 h-4" /> Reveal
            </button>
            <button
              onClick={clear}
              className="inline-flex items-center gap-2 rounded-lg border border-gray-300 bg-white px-4 py-2 text-sm font-medium text-gray-700 hover:bg-gray-50"
            >
              <Eraser className="w-4 h-4" /> Clear
            </button>
          </div>
          {revealed && (
            <p className="text-center text-xs text-gray-400 mt-2">
              Revealed — come back tomorrow for a new puzzle from tomorrow&apos;s headlines.
            </p>
          )}
        </div>

        {/* Clues */}
        <div className="space-y-5">
          {clueLists.map(({ title, items }) => (
            <div key={title}>
              <h3 className="text-sm font-semibold text-gray-700 uppercase tracking-wide mb-2">
                {title}
              </h3>
              <ol className="space-y-1.5">
                {items.map((w) => {
                  const key = `${w.number}:${w.dir}`;
                  const cells = wordPaths[key];
                  const active = cells && selected && cells.some(([r, c]) => r === selected[0] && c === selected[1]);
                  return (
                    <li key={key}>
                      <button
                        onClick={() => {
                          if (cells) focusCell(cells[0][0], cells[0][1]);
                        }}
                        className={`text-left w-full rounded-lg px-3 py-1.5 text-sm transition-colors ${
                          active ? 'bg-blue-50 border border-blue-200' : 'hover:bg-gray-50 border border-transparent'
                        }`}
                      >
                        <span className="font-semibold text-gray-700 mr-2">{w.number}.</span>
                        <span className="text-gray-600">{w.clue}</span>
                        {active && <span className="ml-1 text-[11px] text-blue-500">{w.word.length} letters</span>}
                      </button>
                    </li>
                  );
                })}
              </ol>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
