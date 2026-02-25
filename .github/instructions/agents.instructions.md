---
applyTo:
  - "agents/**"
---
- Follow the Software Engineer Agent v1 spec in `.github/prompts/software-engineer-agent.md`
- Use Python 3.10+ with type hints and PEP8 formatting
- Agent configs use YAML for broad-capability agents, JSON for single-purpose specialists
- Include unit tests for new agent modules in `agents/tests/`
- Follow the escalation protocol for hard blocks before requesting human input
