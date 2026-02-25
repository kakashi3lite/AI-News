# AI-News Agent Handbook

This repository hosts the **AI News Dashboard**, combining a Next.js frontend with a Node/Python backend and extensive MLOps tooling.

## Directory Structure
- `app/` – Next.js application routes and layouts
- `components/` – Reusable React components
- `news/` – Node ingestion and summarisation utilities
- `agents/` – Python agents and utilities
- `mlops/` – Machine learning infrastructure and deployment scripts
- `api/` – OpenAPI specification
- `tests/` – Node and Python tests

## Coding Standards
- **JavaScript**: ES2020+, Prefer functional React components, TailwindCSS for styling
- **Python**: 3.10+, follow PEP8 with type hints

## Build & Test Commands
- `npm run lint` – ESLint via `next lint`
- `npm test` – Node ingestion test suite
- `pytest` – Python test suite

## Prompt Patterns
1. Provide file paths and function names explicitly
2. Break tasks into sequenced subtasks
3. Include before/after examples when modifying code

## Commit & Security Guidelines
- Follow [Conventional Commits](https://www.conventionalcommits.org/) for messages
- Run all linters and tests before committing
- Never commit secrets or credentials; read them from environment variables

## Agents

### Software Engineer Agent v1
- **Config**: `agents/config/swe_agent_config.yaml`
- **Prompt**: `.github/prompts/software-engineer-agent.md`
- **Purpose**: Autonomous, specification-driven engineering agent for AI coding tools
- **Key features**: Zero-confirmation execution, quality gates, escalation protocol, testing pyramid strategy

### Veteran Developer Agent
- **Config**: `agents/config/agent_config.yaml`
- **Implementation**: `agents/veteran_developer_agent.py`
- **CLI**: `agents/cli/agent_cli.py`

### RSE Fetch Specialist
- **Config**: `agents/config/rse_fetch_config.json`
- **Implementation**: `agents/rse_fetch_specialist.py`

### RSE Summarization Engineer
- **Config**: `agents/config/rse_summarization_config.json`
- **Implementation**: `agents/rse_summarization_engineer.py`

### RSE GitHub Integrator
- **Config**: `agents/config/rse_github_config.yaml`
- **Implementation**: `agents/rse_github_integrator.py`
