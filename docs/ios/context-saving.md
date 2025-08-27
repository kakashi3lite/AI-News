# iOS Context Saving (Swift 6) — Secure and Simple

This guide shows how to persist and reuse authenticated context in the AI‑News iOS app using a small Swift Package (`AINewsKit`). It keeps login fast and reliable, minimizes boilerplate, and remains secure by default.

## Principles
- Keychain only: tokens are stored in iOS Keychain (never UserDefaults).
- Single source of truth: a `TokenStore` actor manages the session.
- Auto‑refresh: `APIClient` retries once on 401 using refresh token.
- Simple API: `AuthClient.login()` + `APIClient.request()`.

## Package Layout
```
ios/
  Package.swift
  Sources/AINewsKit/
    Auth/
      AuthClient.swift     # login + refresh
      AuthModels.swift     # token models
      TokenStore.swift     # Keychain‑backed session actor
    Networking/
      APIClient.swift      # authorized requests + 401 refresh
      RequestBuilder.swift # tiny request helper
    Support/
      Keychain.swift       # thin Keychain wrapper
```

## Quick Start
1) Add the local package in Xcode from `ios/`.
2) Configure your base API URL (staging/prod).
3) In your App boot:
```swift
let baseURL = URL(string: "https://your-api-host")!
let tokenStore = TokenStore()
let auth = AuthClient(config: .init(baseURL: baseURL), tokenStore: tokenStore)
let api = APIClient(config: .init(baseURL: baseURL), tokenStore: tokenStore)

// If not logged in, call:
try await auth.login(email: "user@example.com", password: "••••••••")

// Then use the API client everywhere:
struct Article: Decodable { let id: String; let title: String }
let articles: [Article] = try await api.request("/api/v1/news", type: [Article].self)
```

## Security Notes
- Never log tokens; avoid printing request headers.
- Use ATS/TLS; no clear‑text traffic.
- Consider certificate pinning if you control the backend.
- Use BackgroundTasks to refresh tokens when the app resumes or on Wi‑Fi.

## Endpoint Assumptions
- Login: `POST /api/v1/auth/login` → `{ access_token, refresh_token, expires_in }`
- Refresh: `POST /api/v1/auth/refresh` → same shape
- Adjust model fields in `AuthModels.swift` if needed.

## Testing Strategy
- Unit: mock `URLSession` to test `AuthClient` and `APIClient` (200/401/refresh paths).
- UI: XCUITests that verify context reuse (app relaunch still logged in).
- Contract: hit staging with a separate scheme/Config.

This provides a clean, secure foundation for context saving on iOS while staying small and easy to maintain.
