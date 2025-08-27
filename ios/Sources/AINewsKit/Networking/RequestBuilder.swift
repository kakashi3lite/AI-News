import Foundation

public struct RequestBuilder {
    public enum Method: String { case GET, POST, PUT, DELETE, PATCH }

    public static func make(
        baseURL: URL,
        path: String,
        method: Method = .GET,
        headers: [String: String] = [:],
        body: Encodable? = nil
    ) throws -> URLRequest {
        var req = URLRequest(url: baseURL.appending(path: path))
        req.httpMethod = method.rawValue
        headers.forEach { req.setValue($0.value, forHTTPHeaderField: $0.key) }
        if let body {
            req.setValue("application/json", forHTTPHeaderField: "Content-Type")
            req.httpBody = try JSONEncoder().encode(AnyEncodable(body))
        }
        return req
    }
}

private struct AnyEncodable: Encodable {
    private let encodeFunc: (Encoder) throws -> Void
    init(_ base: Encodable) { self.encodeFunc = base.encode }
    func encode(to encoder: Encoder) throws { try encodeFunc(encoder) }
}

