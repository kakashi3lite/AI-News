import { test, expect, anonTest, anonExpect } from './fixtures';

test('homepage renders', async ({ page }) => {
  await page.goto('/');
  await expect(page).toHaveTitle(/AI|News|Dash/i);
});

// Example unauthenticated test (works even if no auth in the app)
anonTest('can navigate to homepage anonymously', async ({ page }) => {
  await page.goto('/');
  await anonExpect(page).toHaveURL(/\/?$/);
});

