export type LoginSelectors = {
  emailTestId: string;
  passwordTestId: string;
  submitTestId: string;
};

export type E2EConfig = {
  baseURL: string;
  loginPath: string;
  postLoginUrlRe: RegExp;
  selectors: LoginSelectors;
};

export function getTestConfig(): E2EConfig {
  const baseURL = process.env.UI_BASE_URL || 'http://localhost:3000';
  const loginPath = process.env.LOGIN_PATH || '/login';
  const postLoginRegex = process.env.POST_LOGIN_REGEX || 'dashboard|home|/$';
  const postLoginUrlRe = new RegExp(postLoginRegex);
  const selectors: LoginSelectors = {
    emailTestId: process.env.EMAIL_TESTID || 'email-input',
    passwordTestId: process.env.PASSWORD_TESTID || 'password-input',
    submitTestId: process.env.SUBMIT_TESTID || 'login-submit',
  };
  return { baseURL, loginPath, postLoginUrlRe, selectors };
}

