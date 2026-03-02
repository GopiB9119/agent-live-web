---
applyTo: "agent/agent/**/*.py,agent/agent/*.md"
---

# Python Agent Instructions

## Goal
Keep MCP connection reliable, tool routing strict, and memory/workspace behavior predictable.

## Rules
- Keep `init_mcp_client` and `shutdown_mcp_client` symmetric and safe.
- On MCP startup failure, always clear session state (`mcp_session`, exit stack).
- Do not claim local paths are inaccessible when fs tools are available.
- Keep tool-call timeout and max-iteration guards configurable via env vars.
- Maintain one-step execute/verify/recover behavior for browser tools.

## Memory rules
- Markdown files are source of truth (`memory/YYYY-MM-DD.md`, `MEMORY.md`).
- Do not store secrets or private credentials in memory logs.
- Keep hybrid memory search behavior explicit and deterministic.

## Validation after edits
- `python -m py_compile agent/agent/tools.py agent/agent/agent.py`
