export type RoleName = 'user' | 'admin' | string;

export type RoleCredentials = {
  email: string;
  password: string;
};

export type RolesConfig = {
  defaultRole: RoleName;
  roles: Record<RoleName, RoleCredentials>;
};

function env(name: string): string | undefined {
  const v = process.env[name];
  return v && v.length > 0 ? v : undefined;
}

export function getRolesConfig(): RolesConfig {
  const defaultRole: RoleName = (env('PW_DEFAULT_ROLE') || 'user');
  const roleList = (env('PW_ROLES') || 'user').split(',').map(s => s.trim()).filter(Boolean);

  const roles: Record<RoleName, RoleCredentials> = {};
  for (const role of roleList) {
    if (role === 'admin') {
      const email = env('TEST_ADMIN_EMAIL') || '';
      const password = env('TEST_ADMIN_PASSWORD') || '';
      if (email && password) roles[role] = { email, password };
    } else if (role === 'user') {
      const email = env('TEST_EMAIL') || '';
      const password = env('TEST_PASSWORD') || '';
      if (email && password) roles[role] = { email, password };
    } else {
      const upper = role.toUpperCase().replace(/[^A-Z0-9]/g, '_');
      const email = env(`TEST_${upper}_EMAIL`) || '';
      const password = env(`TEST_${upper}_PASSWORD`) || '';
      if (email && password) roles[role] = { email, password };
    }
  }

  return { defaultRole, roles };
}

