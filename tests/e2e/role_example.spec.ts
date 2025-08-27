import { expect } from '@playwright/test';
import { useRole } from './fixtures';

// These are illustrative; they will only run meaningfully if the app has roles and routes.
const admin = useRole('admin');
const user = useRole('user');

admin('admin section is visible to admin', async ({ page }) => {
  await page.goto('/admin');
  await expect(page.getByTestId('admin-panel')).toBeVisible();
});

user('admin section blocks user', async ({ page }) => {
  await page.goto('/admin');
  await expect(page.getByTestId('forbidden')).toBeVisible();
});

