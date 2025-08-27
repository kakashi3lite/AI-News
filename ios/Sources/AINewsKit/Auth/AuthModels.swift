import Foundation

public struct AuthTokens: Codable, Sendable {
    public let accessToken: String
    public let refreshToken: String
    public let expiresAt: Date
}

public struct LoginRequest: Codable, Sendable {
    public let email: String
    public let password: String
}

public struct LoginResponse: Codable, Sendable {
    public let access_token: String
    public let refresh_token: String
    public let expires_in: Int
    // user payload omitted for brevity
}

