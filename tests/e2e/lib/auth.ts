import type { Page } from '@playwright/test';
import type { E2EConfig } from '../test.config';

export type Credentials = { email: string; password: string };

export async function performLogin(page: Page, cfg: E2EConfig, creds: Credentials) {
  const { selectors, baseURL, loginPath, postLoginUrlRe } = cfg;
  await page.goto(baseURL + loginPath);
  await page.getByTestId(selectors.emailTestId).fill(creds.email);
  await page.getByTestId(selectors.passwordTestId).fill(creds.password);
  await page.getByTestId(selectors.submitTestId).click();
  await page.waitForURL(postLoginUrlRe, { timeout: 20_000 });
}

