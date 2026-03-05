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
  - Extracted OAuth internals into `agent/agent/oauth_tools.py` and kept API-compatible wrappers in `tools.py`.
  - Extracted memory internals into `agent/agent/memory_tools.py` and kept API-compatible wrappers in `tools.py`.
  - Extracted `web_fetch` internals into `agent/agent/web_tools.py` and kept API-compatible wrapper in `tools.py`.
  - Extracted registry assembly helpers into `agent/agent/tooling/registry.py` and wired `tools.py` to consume them.
  - Extracted MCP lifecycle + browser ownership/retry logic into `agent/agent/mcp_tools.py` with compatibility wrappers in `tools.py`.
  - Extracted filesystem/code-analysis logic into `agent/agent/fs_tools.py` with compatibility wrappers in `tools.py`.
  - Extracted command execution safety/runtime logic into `agent/agent/command_tools.py` with a compatibility wrapper in `tools.py`.
  - Extracted planning/workflow orchestration into `agent/agent/workflow_tools.py` with compatibility wrappers in `tools.py`.
  - Extracted diagnostics utilities (`tool_catalog`, `agent_health_report`) into `agent/agent/diagnostics_tools.py` with compatibility wrappers in `tools.py`.
  - Moved `AGENT_TOOLS` schema source of truth into `agent/agent/tooling/schemas.py`; `tools.py` now imports schemas and focuses on runtime wiring/wrappers.
  - Extracted shared workspace/security/env helpers into `agent/agent/runtime_utils.py` and rewired manager setup in `tools.py`.
  - Added unit tests under `agent/agent/tests/` for `runtime_utils`, `workflow_tools`, and `diagnostics_tools` (13 tests, stdlib `unittest`).
  - Expanded unit coverage for `web_tools`, `memory_tools`, and `mcp_tools` with mocked/local-only tests (total suite now 25 tests).
  - Added CI workflow `.github/workflows/python-agent-tests.yml` to run Python agent tests on push/PR for `agent/**` changes.
- Next:
  - Add a lightweight PR status checklist (health report + tests) and enforce it before release tagging.

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
