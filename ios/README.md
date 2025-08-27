# AINewsKit (Swift Package)

Secure, simple, and smart context saving for the AI‑News iOS app. Provides:
- Keychain‑backed token storage (`TokenStore`)
- Auth client for login/refresh (`AuthClient`)
- API client with auto `Authorization` and one‑time retry on 401 (`APIClient`)

## Install (local)
- In Xcode, add package from local folder `ios/`
- Minimum iOS 17, Swift 6, Swift Concurrency

## Usage
```swift
import AINewsKit

let baseURL = URL(string: "https://your-api-host")!
let tokenStore = TokenStore()
let auth = AuthClient(config: .init(baseURL: baseURL), tokenStore: tokenStore)
let api = APIClient(config: .init(baseURL: baseURL), tokenStore: tokenStore)

// 1) Login once, tokens saved to Keychain
try await auth.login(email: "user@example.com", password: "••••••••")

// 2) Make authorized request; client attaches access token and refreshes if needed
struct Article: Decodable { let id: String; let title: String }
let articles: [Article] = try await api.request("/api/v1/news", type: [Article].self)
```

## Security
- Tokens are only stored in Keychain (never UserDefaults)
- No token logs; avoid printing tokens
- ATS/TLS required by default (configure Info.plist if needed)

## Notes
- Endpoints assume your FastAPI auth routes (`/api/v1/auth/login`, `/refresh`) found in this repo’s backend.
- Adjust models if your payload differs (AuthModels.swift).
- Consider BackgroundTasks to refresh tokens periodically and prefetch content.
