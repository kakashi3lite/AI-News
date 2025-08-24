# Project Brief

**AI News Dashboard** is a full‑stack platform that ingests news, summarises content with AI models and presents results through a Next.js interface.

## Domain & Purpose
- Aggregates RSS/API news sources and normalises entries
- Generates multi‑model summaries via `/summarize` API
- Serves personalised dashboards to end users

## Key Flows
1. **Ingestion** – `news/ingest.js` pulls and deduplicates articles
2. **Summarisation** – POST requests to `/summarize` or `/summarize/batch` produce AI summaries
3. **Presentation** – React components render feeds and analytics

## Public Interfaces
- REST API defined in `api/openapi.yaml`
- JSON schema for stored entries in `schemas/news-entry.schema.json`

## Environments
- Node 18 runtime for frontend and ingestion
- Python 3.10+ for agents and MLOps scripts
- Development server: `npm run dev`
- API server: `npm run dev` + backend scripts as needed
