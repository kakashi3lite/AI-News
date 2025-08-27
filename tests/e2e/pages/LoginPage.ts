import type { Page } from '@playwright/test';
import type { E2EConfig } from '../test.config';

export class LoginPage {
  constructor(private readonly page: Page, private readonly cfg: E2EConfig) {}

  async goto() {
    await this.page.goto(this.cfg.baseURL + this.cfg.loginPath);
  }

  async login(email: string, password: string) {
    const { emailTestId, passwordTestId, submitTestId } = this.cfg.selectors;
    await this.page.getByTestId(emailTestId).fill(email);
    await this.page.getByTestId(passwordTestId).fill(password);
    await this.page.getByTestId(submitTestId).click();
  }
}

