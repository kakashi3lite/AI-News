import { test as base } from '@playwright/test';
import { storagePathFor } from './lib/storage';

export const test = base;
export const expect = test.expect;

// Anonymous fixture: always starts without saved storage
export const anonTest = base.extend({
  context: async ({ browser }, use) => {
    const context = await browser.newContext({ storageState: undefined });
    await use(context);
    await context.close();
  },
  page: async ({ context }, use) => {
    const page = await context.newPage();
    await use(page);
  },
});

export const anonExpect = anonTest.expect;

// Factory to create a test bound to a role's storage state
export function useRole(role: string) {
  const roleStorage = storagePathFor(role);
  return base.extend({
    context: async ({ browser }, use) => {
      const context = await browser.newContext({ storageState: roleStorage });
      await use(context);
      await context.close();
    },
    page: async ({ context }, use) => {
      const page = await context.newPage();
      await use(page);
    },
  });
}

