import Foundation

public actor TokenStore {
    public enum StoreError: Error { case encodingFailed, decodingFailed }

    private let keychainItem = KeychainItem(service: "com.ainews.auth", account: "session")
    private var cached: AuthTokens?

    public init() {}

    public func load() throws -> AuthTokens? {
        if let cached { return cached }
        guard let data = try Keychain.get(for: keychainItem) else { return nil }
        guard let tokens = try? JSONDecoder().decode(AuthTokens.self, from: data) else { throw StoreError.decodingFailed }
        cached = tokens
        return tokens
    }

    public func save(_ tokens: AuthTokens) throws {
        let data = try JSONEncoder().encode(tokens)
        try Keychain.set(data, for: keychainItem)
        cached = tokens
    }

    public func clear() throws {
        try Keychain.remove(for: keychainItem)
        cached = nil
    }

    public func isAccessTokenValid(leeway seconds: TimeInterval = 30) async throws -> Bool {
        guard let t = try await load() else { return false }
        return t.expiresAt.timeIntervalSinceNow > seconds
    }
}

