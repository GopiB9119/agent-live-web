# Python Agent Architecture Plan

## Goal
Keep the Python agent maintainable as features grow (models, MCP tools, OAuth, memory, automation) without turning `agent.py` and `tools.py` into fragile monoliths.

## Current Risks
- `agent/agent/tools.py` is very large and mixes multiple concerns (filesystem, memory, web, OAuth, MCP, registry).
- New tools can be added quickly, but long-term readability and testability degrade.
- Runtime behavior is improving fast; architecture must keep up to avoid regressions.

## Target Module Layout
Phase target layout (incremental, non-breaking):

```text
agent/agent/
  agent.py                    # conversation loop + command handling only
  SYSTEM_PROMPT.md
  architecture/
    config.py                 # env/config/model-provider loading
    runtime.py                # session state + context trimming + tool execution helpers
  tooling/
    registry.py               # AGENT_TOOLS + AVAILABLE_FUNCTIONS registration and validation
    fs_tools.py               # file system tools
    memory_tools.py           # memory tools + vector index logic
    web_tools.py              # web_fetch + URL/SSRF helpers
    oauth_tools.py            # OAuth profile/token management
    mcp_tools.py              # MCP initialization/wrappers/tab ownership checks
```

## Refactor Phases
1. Safety net first
- Keep `agent_health_report` in CI/manual checks before and after refactors.
- Add small behavior tests for high-risk tools (memory_search, web_fetch, oauth_get_token, mcp init path).

2. Extract pure helpers
- Move utility helpers with no runtime side-effects into dedicated modules.
- Keep public function names unchanged to avoid prompt/tool-call breakage.

3. Split tool domains
- Move tools by domain into `tooling/*_tools.py`.
- Re-export from a registry layer so the model-facing schema remains stable.

4. Stabilize contracts
- Add a canonical schema source of truth (single registry builder).
- Validate no duplicate tool names and no schema/callable drift.

5. Performance + testability
- Add unit tests for tool routing and report-format consistency.
- Add smoke tests for provider selection (OpenAI/Azure) and MCP connect fallback behavior.

## Phase Status (Current)
- Completed:
  - Added `agent_health_report` tool for schema/registry/size integrity checks.
  - Added interactive `/doctor` command in `agent.py`.
  - Extracted all domain modules: `oauth_tools.py`, `memory_tools.py`, `web_tools.py`, `mcp_tools.py`, `fs_tools.py`, `command_tools.py`, `workflow_tools.py`, `diagnostics_tools.py`.
  - Moved `AGENT_TOOLS` schema source of truth into `tooling/schemas.py`; `tools.py` now imports schemas and focuses on runtime wiring.
  - Extracted shared workspace/security/env helpers into `runtime_utils.py`.
  - Comprehensive unit test coverage: 87 Python tests across 11 test files + 3 opt-in live MCP integration tests.
  - 44 JS unit tests covering EdgeSession actions, verification, redaction, NL parser chain, and response contracts.
  - CI workflow runs full JS + Python test suite on push/PR.
  - Security hardening: AST-based calculator (no eval), PowerShell metachar injection blocking, SSRF DNS resolution checks.
  - `time.perf_counter()` for accurate workflow duration measurement.
  - Project docs organized into `docs/` directory; root kept clean.
  - `.devcontainer` configured for Codespaces onboarding.
- Next:
  - Wire a scheduled/nightly workflow for live MCP integration runs and artifact upload.
  - Extract `agent.py` config/runtime helpers into `architecture/config.py` and `architecture/runtime.py` (Phase 2 target).

## Engineering Rules
- One tool = one clear responsibility.
- Keep side effects explicit and audited (delete/send/submit).
- Prefer additive refactors; keep old interfaces until migration completes.
- Never break existing tool names without adding compatibility aliases.
- Run `agent_health_report` after any tool schema or registry change.

## Operational Checklist
- Before merge:
  - `agent_health_report` returns `ok` or acceptable `warn`.
  - No duplicate tool schema names.
  - Syntax check passes for `agent.py` and `tools.py`.
  - README and SYSTEM_PROMPT reflect any new tools/capabilities.
