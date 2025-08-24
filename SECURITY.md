# Security

## ASVS L2 Checklist
| Control | Status | Notes |
|---------|--------|-------|
| Authentication | Gap | API spec notes API key but no session management |
| Access Control | Gap | No role-based checks in repo |
| Input Validation | Pass | JSON schemas validate news entries |
| Output Encoding | Gap | Frontend rendering lacks explicit encoding guidance |
| Error Handling | Pass | Tests verify ingestion error paths |
| Logging & Monitoring | Gap | Minimal centralized logging |
| Secrets Management | Gap | `.env` usage implied but not enforced |
| Dependency Management | Pass | `npm audit` in CI workflow |
| Transport Security | N/A | SSL termination handled by hosting |
| Secure Headers/CSP | Gap | No CSP configuration detected |
| Deserialization | Pass | Uses JSON parsing only |
| SSRF Protection | Gap | Ingestion fetches external URLs without allowlist |
| Rate Limiting | Pass | `/summarize` endpoints specify limits |

## Secret Handling
- Never commit API keys or credentials
- Use environment variables and secret stores

## Dependency Risks
- Monitor `axios`, `ytdl-core`, and Python packages for CVEs
