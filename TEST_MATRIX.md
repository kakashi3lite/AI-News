# Test Matrix

| Area | Framework | Existing Tests | Coverage | Gaps |
|------|-----------|----------------|----------|------|
| News ingestion (Node) | Custom JS | `tests/test_ingest.js` | Moderate | Lacks network failure tests |
| Agents & Python utils | pytest | `agents/tests/*.py`, `test_rse_content.py` | Low | No tests for `mlops/` modules |
| Frontend components | n/a | none | 0% | Missing React component tests |
| API contract | n/a | none | 0% | No tests for OpenAPI endpoints |
| Performance | Node script | `npm run test:benchmark` | Manual | Automated regression missing |

## Recommended Tests
1. Jest tests for core React components
2. Integration tests hitting `/summarize` endpoints
3. Python unit tests for `mlops/` orchestration
4. Error-path tests for ingestion engine
5. Smoke tests for deployment scripts
