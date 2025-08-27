import { defineConfig, devices } from '@playwright/test';

const isCI = !!process.env.CI;
const baseURL = process.env.UI_BASE_URL || 'http://localhost:3000';

export default defineConfig({
  testDir: './e2e',
  fullyParallel: true,
  forbidOnly: isCI,
  retries: isCI ? 2 : 0,
  workers: process.env.PW_WORKERS ? Number(process.env.PW_WORKERS) : (isCI ? 2 : undefined),
  reporter: isCI
    ? [ ['list'], ['github'], ['junit', { outputFile: 'test-results/junit.xml' }], ['html', { open: 'never' }], ['blob'] ]
    : [ ['list'], ['html', { open: 'never' }] ],
  use: {
    baseURL,
    trace: 'on-first-retry',
    screenshot: 'only-on-failure',
    video: isCI ? 'retain-on-failure' : 'off',
    headless: process.env.HEADLESS ? process.env.HEADLESS === '1' : isCI,
  },
  // Auto-start Next.js dev server for E2E (adjust if using preview/build)
  webServer: process.env.PW_DISABLE_WEBSERVER === '1' ? undefined : {
    command: 'npm run dev',
    url: baseURL,
    reuseExistingServer: !isCI,
    timeout: 90_000,
  },
  projects: [
    { name: 'chromium', use: { ...devices['Desktop Chrome'] } },
    { name: 'firefox', use: { ...devices['Desktop Firefox'] } },
    { name: 'webkit', use: { ...devices['Desktop Safari'] } },
  ],
  globalSetup: require.resolve('./e2e/auth.setup'),
  outputDir: 'test-results',
});

