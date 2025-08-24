# Architecture

## Component Overview
- **Next.js Frontend** (`app/`, `components/`) serves dashboard UI
- **Ingestion Engine** (`news/ingest.js`) collects and deduplicates articles
- **Summarization API** (`api/openapi.yaml`) exposes `/summarize` endpoints
- **MLOps Services** (`mlops/`) manage training, deployment, and monitoring

## High-Level Diagram
```mermaid
graph TD
  User -->|browse| Frontend
  Frontend -->|fetch| SummarizationAPI
  Frontend -->|ingest triggers| IngestionEngine
  IngestionEngine -->|stores| DataStore[(News Store)]
  SummarizationAPI -->|reads| DataStore
  SummarizationAPI -->|delegates| MLServices
```

## Summarization Sequence
```mermaid
sequenceDiagram
  participant U as User
  participant F as Frontend
  participant A as API
  participant M as Model
  U->>F: request summary
  F->>A: POST /summarize
  A->>M: generate summary
  M-->>A: summary text
  A-->>F: JSON response
  F-->>U: render summary
```

## Deployment Topology
- Frontend deployed on Vercel or Node server
- Backend and MLOps services containerized via Docker
- Optional Kubernetes manifests under `mlops/kubernetes/`
