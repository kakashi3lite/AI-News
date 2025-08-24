# Benchmarks

`tests/test_ingest.js` includes a benchmarking harness measuring RSS parsing, classification, and deduplication times.

## Targets
- RSS parsing < 50ms avg
- Topic classification < 200ms for 100 articles
- Deduplication < 100ms for 100 articles

## Running
```bash
npm run test:benchmark
```
