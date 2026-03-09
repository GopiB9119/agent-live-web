---
applyTo: "agent/agent/**/*.py,agent/agent/*.md"
---

# Python Agent Instructions

## Goal
Keep the primary Python agent surface reliable, keep tool routing strict, and keep memory/workspace behavior predictable.

## Rules
- Keep `init_mcp_client` and `shutdown_mcp_client` symmetric and safe.
- On MCP startup failure, always clear session state (`mcp_session`, exit stack).
- Do not claim local paths are inaccessible when fs tools are available.
- Keep tool-call timeout and max-iteration guards configurable via env vars.
- Maintain one-step execute/verify/recover behavior for browser tools.
- Keep preview / confirm / blocked policy behavior aligned with `docs/SAFETY_GATING_DESIGN.md`.
- Treat browser/MCP calls as tool infrastructure under the Python control plane, not as a separate product path.

## Memory rules
- Markdown files are source of truth (`memory/YYYY-MM-DD.md`, `MEMORY.md`).
- Do not store secrets or private credentials in memory logs.
- Keep hybrid memory search behavior explicit and deterministic.

## Validation after edits
- `python -m py_compile agent/agent/tools.py agent/agent/agent.py`
- `npm run agent:test:py`
