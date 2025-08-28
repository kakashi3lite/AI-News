import Foundation

public struct EnvironmentSession: Codable, Sendable {
    public let tokens: AuthTokens
    public var metadata: SessionMetadata
}

public actor EnvironmentScopedTokenStore {
    public enum StoreError: Error { case encodingFailed, decodingFailed }

    private static func envKey(from url: URL) -> String {
        if let host = url.host { return host + (url.port != nil ? ":\(url.port!)" : "") }
        return url.absoluteString
    }

    private func keychainItem(envKey: String) -> KeychainItem {
        KeychainItem(service: "com.ainews.auth.env", account: "session-\(envKey)")
    }

    private var cache: [String: EnvironmentSession] = [:]

    public init() {}

    // Migration: pull legacy unscoped session if present
    public func migrateIfNeeded(fromLegacy legacyStore: TokenStore, baseURL: URL) async throws {
        let env = Self.envKey(from: baseURL)
        if cache[env] != nil { return }
        let item = keychainItem(envKey: env)
        if let _ = try Keychain.get(for: item) { return } // already present

        if let legacy = try await legacyStore.load() {
            let meta = SessionMetadata(envKey: env)
            let envSession = EnvironmentSession(tokens: legacy, metadata: meta)
            try await save(envSession, forEnv: env)
            try await legacyStore.clear()
        }
    }

    public func load(for baseURL: URL) throws -> EnvironmentSession? {
        let env = Self.envKey(from: baseURL)
        if let c = cache[env] { return c }
        let item = keychainItem(envKey: env)
        guard let data = try Keychain.get(for: item) else { return nil }
        guard let envSession = try? JSONDecoder().decode(EnvironmentSession.self, from: data) else { throw StoreError.decodingFailed }
        cache[env] = envSession
        return envSession
    }

    public func save(tokens: AuthTokens, for baseURL: URL, claimsHash: String? = nil) async throws {
        let env = Self.envKey(from: baseURL)
        var meta = (try load(for: baseURL))?.metadata ?? SessionMetadata(envKey: env)
        meta.lastValidatedAt = Date()
        meta.claimsHash = claimsHash
        let envSession = EnvironmentSession(tokens: tokens, metadata: meta)
        try await save(envSession, forEnv: env)
    }

    private func save(_ envSession: EnvironmentSession, forEnv env: String) async throws {
        let data = try JSONEncoder().encode(envSession)
        try Keychain.set(data, for: keychainItem(envKey: env))
        cache[env] = envSession
    }

    public func clear(for baseURL: URL) throws {
        let env = Self.envKey(from: baseURL)
        try Keychain.remove(for: keychainItem(envKey: env))
        cache[env] = nil
    }
}
