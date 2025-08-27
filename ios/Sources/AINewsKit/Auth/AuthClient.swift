import Foundation

public struct AuthClient: Sendable {
    public struct Config: Sendable {
        public let baseURL: URL
        public init(baseURL: URL) { self.baseURL = baseURL }
    }

    private let config: Config
    private let tokenStore: TokenStore
    private let session: URLSession

    public init(config: Config, tokenStore: TokenStore, session: URLSession = .shared) {
        self.config = config
        self.tokenStore = tokenStore
        self.session = session
    }

    public func login(email: String, password: String) async throws {
        let url = config.baseURL.appending(path: "/api/v1/auth/login")
        var req = URLRequest(url: url)
        req.httpMethod = "POST"
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        let body = ["email": email, "password": password]
        req.httpBody = try JSONSerialization.data(withJSONObject: body)

        let (data, resp) = try await session.data(for: req)
        guard let http = resp as? HTTPURLResponse, (200..<300).contains(http.statusCode) else {
            throw URLError(.badServerResponse)
        }
        let login = try JSONDecoder().decode(LoginResponse.self, from: data)

        let expiresAt = Date().addingTimeInterval(TimeInterval(login.expires_in))
        let tokens = AuthTokens(accessToken: login.access_token, refreshToken: login.refresh_token, expiresAt: expiresAt)
        try await tokenStore.save(tokens)
    }

    public func refreshIfNeeded() async throws -> String? {
        if try await tokenStore.isAccessTokenValid() { return try await tokenStore.load()?.accessToken }
        guard let current = try await tokenStore.load() else { return nil }
        var req = URLRequest(url: config.baseURL.appending(path: "/api/v1/auth/refresh"))
        req.httpMethod = "POST"
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        let body = ["refresh_token": current.refreshToken]
        req.httpBody = try JSONSerialization.data(withJSONObject: body)

        let (data, resp) = try await session.data(for: req)
        guard let http = resp as? HTTPURLResponse, (200..<300).contains(http.statusCode) else {
            throw URLError(.badServerResponse)
        }
        let refreshed = try JSONDecoder().decode(LoginResponse.self, from: data)
        let expiresAt = Date().addingTimeInterval(TimeInterval(refreshed.expires_in))
        let tokens = AuthTokens(accessToken: refreshed.access_token, refreshToken: refreshed.refresh_token, expiresAt: expiresAt)
        try await tokenStore.save(tokens)
        return tokens.accessToken
    }
}

