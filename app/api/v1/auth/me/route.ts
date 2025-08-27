export async function GET(request: Request): Promise<Response> {
  const auth = request.headers.get('authorization') || '';
  const token = auth.replace(/^Bearer\s+/i, '').trim();
  const isAdmin = token.toLowerCase().includes('admin');
  const roles = isAdmin ? ['admin'] : ['user'];
  const permissions = isAdmin ? ['admin:all'] : ['user:read'];
  return new Response(JSON.stringify({ roles, permissions, claims_version: 1 }), {
    headers: { 'content-type': 'application/json' },
    status: 200,
  });
}

