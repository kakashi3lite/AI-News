import Foundation

public struct SessionMetadata: Codable, Sendable {
    public static let schemaVersion = 1

    public let schema: Int
    public let envKey: String
    public let createdAt: Date
    public var lastValidatedAt: Date
    public var claimsHash: String?

    public init(envKey: String, createdAt: Date = Date(), lastValidatedAt: Date? = nil, claimsHash: String? = nil) {
        self.schema = Self.schemaVersion
        self.envKey = envKey
        self.createdAt = createdAt
        self.lastValidatedAt = lastValidatedAt ?? createdAt
        self.claimsHash = claimsHash
    }
}

public enum ClaimsHasher {
    public static func hash(roles: [String], permissions: [String] = []) -> String {
        let sorted = (roles.map { $0.lowercased() } + permissions.map { $0.lowercased() }).sorted()
        let joined = sorted.joined(separator: "|")
        #if canImport(CryptoKit)
        import CryptoKit
        let digest = SHA256.hash(data: Data(joined.utf8))
        return Data(digest).base64EncodedString()
        #else
        // Fallback simple hash (not cryptographic, but stable)
        return String(joined.hashValue)
        #endif
    }
}

