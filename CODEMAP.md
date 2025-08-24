# Code Map

```
root/
├── app/                  # Next.js routes (owner: @frontend-team)
├── components/           # React UI components (@frontend-team)
├── news/                 # Node ingestion utilities (@backend-team)
├── agents/               # Python automation agents (@mlops-team)
├── mlops/                # ML infrastructure & deployment (@mlops-team)
├── api/openapi.yaml      # REST API specification
├── schemas/              # JSON schemas for news entries
├── tests/                # Node & Python tests
└── scripts/              # Helper scripts
```

## Entrypoints & Hot Paths
- `news/ingest.js` – ingestion engine used by tests and CLI
- `app/layout.js` & `app/page.js` – Next.js entrypoints
- `mlops/orchestrator.py` – orchestrates ML workflows

## Data Flow
News sources → `news/ingest.js` → `schemas/news-entry.schema.json` → summaries via `/summarize` API → rendered by `components/` and `app/`.
