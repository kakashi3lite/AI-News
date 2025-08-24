# API Catalog

| Endpoint | Method | Description | Auth | Rate Limit |
|----------|--------|-------------|------|------------|
| `/summarize` | POST | Summarize single article or text | None | 100/hr standard |
| `/summarize/batch` | POST | Summarize multiple articles | None | 50 articles/request |
| `/summarize/models` | GET | List available models | None | - |
| `/summarize/health` | GET | Service health check | None | - |
| `/summarize/cache` | DELETE | Clear cache (admin) | API key | - |

## Request/Response Schemas
- `SummarizeRequest` and `SummarizeResponse` in `api/openapi.yaml`
- Batch variants `BatchSummarizeRequest` and `BatchSummarizeResponse`
