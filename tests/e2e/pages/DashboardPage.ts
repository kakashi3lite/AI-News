import type { Page } from '@playwright/test';

export class DashboardPage {
  constructor(private readonly page: Page) {}

  async expectLoaded() {
    await this.page.waitForLoadState('networkidle');
  }
}

