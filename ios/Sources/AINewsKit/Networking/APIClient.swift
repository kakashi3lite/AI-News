import Foundation

public final class APIClient: @unchecked Sendable {
    public struct Config: Sendable {
        public let baseURL: URL
        public init(baseURL: URL) { self.baseURL = baseURL }
    }

    private let config: Config
    private let session: URLSession
    private let tokenStore: TokenStore
    private let authClient: AuthClient

    public init(config: Config, session: URLSession = .shared, tokenStore: TokenStore) {
        self.config = config
        self.session = session
        self.tokenStore = tokenStore
        self.authClient = AuthClient(config: .init(baseURL: config.baseURL), tokenStore: tokenStore, session: session)
    }

    public func request<T: Decodable>(
        _ path: String,
        method: RequestBuilder.Method = .GET,
        headers: [String: String] = [:],
        body: Encodable? = nil,
        authorized: Bool = true,
        type: T.Type
    ) async throws -> T {
        var hdrs = headers
        if authorized, let token = try await tokenStore.load()?.accessToken {
            hdrs["Authorization"] = "Bearer \(token)"
        }
        var req = try RequestBuilder.make(baseURL: config.baseURL, path: path, method: method, headers: hdrs, body: body)

        let (data, resp) = try await session.data(for: req)
        if let http = resp as? HTTPURLResponse, http.statusCode == 401, authorized {
            // Try refresh once
            if let newToken = try await authClient.refreshIfNeeded() {
                req.setValue("Bearer \(newToken)", forHTTPHeaderField: "Authorization")
                let (data2, resp2) = try await session.data(for: req)
                guard let http2 = resp2 as? HTTPURLResponse, (200..<300).contains(http2.statusCode) else {
                    throw URLError(.userAuthenticationRequired)
                }
                return try JSONDecoder().decode(T.self, from: data2)
            }
        }
        guard let http = resp as? HTTPURLResponse, (200..<300).contains(http.statusCode) else {
            throw URLError(.badServerResponse)
        }
        return try JSONDecoder().decode(T.self, from: data)
    }
}

