# iOS Context Saving Checkpoints (Swift 6)

This guide defines practical checkpoints to keep authentication context reliable, secure, and fast across app lifecycle events. It builds on AINewsKit’s TokenStore and EnvironmentScopedTokenStore.

## Goals
- Keep users signed in seamlessly without re-entering credentials.
- Ensure tokens are valid and scoped to the current environment (host).
- Detect permission/role drift and rotate sessions accordingly.
- Fail safe: no token leaks; no reliance on UserDefaults.

## Checkpoints

1) App Launch Checkpoint
- Load env-scoped session: `EnvironmentScopedTokenStore.load(for: baseURL)`.
- Validate expiry: if expiring soon, refresh; else proceed.
- Optional drift check: fetch `/api/v1/auth/me`, compute `claimsHash`, compare to metadata; if changed → refresh or re-login.
- Route decision: present Login when no valid session.

Snippet:
```swift
@main
struct AppMain: App {
  @Environment(\.scenePhase) private var scenePhase
  let baseURL = URL(string: AppConfig.apiBaseURL)! // host defines envKey
  let envStore = EnvironmentScopedTokenStore()
  let tokenStore = TokenStore() // legacy
  let auth: AuthClient
  let api: APIClient

  init() {
    try? awaitTask { try await envStore.migrateIfNeeded(fromLegacy: tokenStore, baseURL: baseURL) }
    auth = AuthClient(config: .init(baseURL: baseURL), tokenStore: tokenStore)
    api = APIClient(config: .init(baseURL: baseURL), tokenStore: tokenStore)
  }

  var body: some Scene { WindowGroup { RootView(auth: auth, api: api, envStore: envStore, baseURL: baseURL) } }
}
```

2) Pre-flight Request Checkpoint
- Ensure valid access token before requests.
- Auto-refresh on 401 exactly once; fail with a consistent error type otherwise.

Pattern:
```swift
let articles: [Article] = try await api.request("/api/v1/news", type: [Article].self)
```
(APIClient attaches Bearer and retries once on 401.)

3) Foreground Resume Checkpoint
- On `scenePhase == .active`, re-validate if last validation > N minutes or if app was backgrounded long.

Snippet:
```swift
.onChange(of: scenePhase) { _, phase in
  guard phase == .active else { return }
  Task { try? await validateSession() }
}

func validateSession() async throws {
  if try await tokenStore.isAccessTokenValid() { return }
  _ = try await auth.refreshIfNeeded() // rethrows on failure
}
```

4) Background Refresh Checkpoint
- Schedule `BGAppRefreshTask` to refresh tokens and prefetch above-the-fold data on Wi‑Fi/charging.

Checklist:
- Register task identifier in Info.plist.
- Submit tasks after successful login and on app exit.
- Respect battery/network constraints; exponential backoff on failures.

5) Critical Action Checkpoint
- Before sensitive operations (payments, profile updates), ensure freshness: if token older than threshold, refresh proactively.
- For high-risk flows, require interactive re-auth (server policy driven).

6) Logout/Revocation Checkpoint
- Call backend logout endpoint if available; blacklist token.
- Clear Keychain via `envStore.clear(for:)` and wipe sensitive caches.
- Route to Login; avoid race conditions with in-flight requests.

7) Migration Checkpoint
- On app upgrades, run `migrateIfNeeded(fromLegacy:baseURL:)` once to move legacy session to env-scoped storage.
- Increment metadata schema when storage format changes; migrate accordingly.

8) Multi-Environment Checkpoint
- Derive `envKey` from `URL.host[:port]`.
- Never reuse tokens across `localhost`, `staging`, `prod`.
- Maintain separate sessions per environment in Keychain.

9) Error Handling Checkpoint
- Normalize errors: `URLError.userAuthenticationRequired` for auth failures, retryable vs fatal categories.
- Add light user messaging for expired sessions; avoid exposing internals.

10) Telemetry Checkpoint (Local, No PII)
- Counters: refresh attempts/successes, 401 rates, drift events.
- Don’t log tokens, headers, or user identifiers; prefer aggregate metrics.

11) Testing Checkpoints
- Unit: TokenStore save/load/clear; EnvironmentScoped migration; ClaimsHasher stability.
- Integration: mock `/login`, `/refresh`, `/me` to exercise refresh and drift.
- UI: XCUITests across cold/warm launches; offline/online toggles.

## Recommended Defaults
- Keychain accessibility: `kSecAttrAccessibleAfterFirstUnlockThisDeviceOnly`.
- Refresh window: refresh if `expiresAt - now < 30s`.
- Drift cadence: check `/me` at launch and after 6h foreground interval.
- Background refresh interval: minimum allowed by OS; don’t fight iOS scheduler.

## Example Integration
```swift
final class AppState: ObservableObject {
  let baseURL: URL
  let auth: AuthClient
  let api: APIClient
  let envStore = EnvironmentScopedTokenStore()
  let tokenStore = TokenStore()
  @Published var isAuthenticated = false

  init(baseURL: URL) {
    self.baseURL = baseURL
    self.auth = AuthClient(config: .init(baseURL: baseURL), tokenStore: tokenStore)
    self.api = APIClient(config: .init(baseURL: baseURL), tokenStore: tokenStore)
    Task { await bootstrap() }
  }

  private func bootstrap() async {
    try? await envStore.migrateIfNeeded(fromLegacy: tokenStore, baseURL: baseURL)
    self.isAuthenticated = (try? await tokenStore.isAccessTokenValid()) ?? false
  }

  func login(email: String, password: String) async throws {
    try await auth.login(email: email, password: password)
    self.isAuthenticated = true
  }

  func refreshIfNeeded() async throws {
    _ = try await auth.refreshIfNeeded()
  }

  func logout() async {
    try? await envStore.clear(for: baseURL)
    self.isAuthenticated = false
  }
}
```

## Runbook (Common Issues)
- “User logged out unexpectedly”: look for drift events or refresh failures → prompt login.
- “401 loops”: ensure APIClient retries once only; don’t infinite-loop on refresh.
- “Staging tokens used in prod”: confirm envKey derivation and per-env storage in Keychain.
- “Tokens lost on device reboot”: verify Keychain accessibility attribute.

## Security Notes
- Keychain only; never UserDefaults or plaintext files.
- ATS/TLS enforced; consider certificate pinning on critical endpoints.
- No secrets in logs or analytics.

## References
- AINewsKit: `EnvironmentScopedTokenStore`, `SessionMetadata`, `TokenStore`, `AuthClient`, `APIClient`.
- Backend: `/api/v1/auth/login`, `/api/v1/auth/refresh`, `/api/v1/auth/me` (whoami for drift).
