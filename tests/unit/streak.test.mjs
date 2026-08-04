import { test } from 'node:test';
import assert from 'node:assert/strict';

const { nextStreak } = await import('../../lib/client/userStore.js');

test('first solve starts a 1-day streak', () => {
  assert.deepEqual(nextStreak(null, 0, '2026-08-05'), { streak: 1, sameDay: false });
});

test('solving again the same day does not increment', () => {
  assert.deepEqual(nextStreak('2026-08-05', 3, '2026-08-05'), { streak: 3, sameDay: true });
});

test('solving the next day increments the streak', () => {
  assert.deepEqual(nextStreak('2026-08-04', 3, '2026-08-05'), { streak: 4, sameDay: false });
});

test('a missed day resets the streak to 1', () => {
  assert.deepEqual(nextStreak('2026-08-02', 5, '2026-08-05'), { streak: 1, sameDay: false });
});
