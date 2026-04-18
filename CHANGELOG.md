# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [6.0.0] - 2026-04-06

### Breaking Changes
- Removed `agent-logic.js` (deprecated since v5.2). Use `const { EdgeSession } = require('agent-works')`.
- Removed `playwright-edge-agent.js` (deprecated since v5.2). Use `npm run agent:live-web`.

### Features
- **Git tools**: `git_status`, `git_diff`, `git_log`, `git_blame`, `git_commit`, `git_branch`, `git_stash` — native git operations without shelling out.
- **Test generation**: `generate_tests`, `run_tests`, `coverage_gaps` — auto-detect framework, generate test skeletons, analyze coverage.
- **Snapshot/rollback**: `snapshot_create`, `snapshot_restore`, `snapshot_list`, `snapshot_diff` — undo system for file edits.
- **Refactoring tools**: `rename_symbol`, `find_dead_code`, `find_duplicates`, `code_metrics` — code quality analysis and transformation.
- **Vision tools**: `vision_encode`, `vision_compare`, `vision_describe_page` — screenshot analysis for LLM vision models.
- **Documentation tools**: `generate_docstrings`, `generate_changelog_entry`, `doc_coverage` — auto-generate docs and measure coverage.
- Extracted `config.py` from `agent.py` — cleaner separation of configuration from conversation loop.
- Added nightly integration CI workflow (`nightly-integration.yml`).
- CI now runs full JS + Python test suite on every push/PR.
- Added `tracing.js` to syntax check script.
- Proper `.devcontainer/devcontainer.json` for Codespaces onboarding.

### Security
- Replaced `eval()` with AST-based expression evaluator in `calculate()` — eliminates code injection risk.
- Added PowerShell metacharacter injection blocking (`$()`, backtick, `&{}`, `@()`, `--%`, `Invoke-Expression`).
- 0 npm audit vulnerabilities across 157 dependencies.

### Bug Fixes
- Fixed missing `await` on `this.getLocator()` in `handleSelect()` and `handleClear()` — prevented runtime crash.
- Fixed test mocks using sync functions for async `getLocator()` — tests now properly validate async correctness.
- Fixed `TimeoutExpired` constructor compatibility with Python 3.14.

### Improvements
- `time.time()` → `time.perf_counter()` in workflow duration measurement.
- Removed duplicate npm scripts (`agent:test:py`, `agent:test:py:integration`).
- Moved 4 docs from root to `docs/` directory.
- Added `.gitignore` rules for generated eval pipeline output.
- Updated `ARCHITECTURE_PLAN.md` to reflect current state (58 tools, 136 tests).
- Filled `package.json` author field.
- Updated `SECURITY.md` supported versions table.

### Stats
- **58 callable tools** (was 34 in v5.2)
- **136 tests** passing (44 JS + 92 Python)
- **0 vulnerabilities**
- **MCP preflight**: 8/8 passing
- **Governance**: ALL PASS

## [5.2.0] - 2026-03-28

### Features
- npm entry point `index.js` exposing `EdgeSession`, `parseCommand`, tracing.
- `.env.example` with 40+ documented env vars.
- `scripts/resolve-python.js` — auto-finds venv Python for npm scripts.
- `scripts/mcp-preflight.js` — 8 automated Edge runtime checks.
- `scripts/validate-web-task.js` — validates task JSON against schema.
- New browser actions: back, forward, refresh, press (NL parser + EdgeSession + verification).
- Recursion guards: `call_tool` blocks orchestration tools, `workflow_execute` blocks `call_tool`/`task_autopilot`.
- `npm test` now runs real tests (not just syntax check).
- `test:all` runs JS + Python in one command.
- Web-task examples in `.github/skills/web-works/examples/`.
- `MEMORY.md` seeded with project reference data.

### Security
- Restricted command execution by default (`AGENT_RUN_COMMAND_SECURITY_MODE=restricted`).
- SSRF protection with DNS-resolving `is_private_or_local_host`.
- Memory auto-log disabled by default.
- Sensitive data redaction in logs, traces, memory, and tool output.

## [5.1.1] - 2026-03-20

### Features
- Modularized Python agent: extracted 9 domain modules from monolithic `tools.py`.
- Added `tooling/schemas.py` as single source of truth for tool definitions.
- Added `tooling/registry.py` for auto-registration.
- CI workflow for Python agent tests.
- Initial v5.2 MCP integration scaffolding.
- Health snapshot reporting with artifact upload.

### Security
- Hardened command execution and web fetch.
- Added memory log redaction.
- Cleaned tracked local artifacts from git.

## [5.1.0] - 2026-03-15

### Features
- Full evaluation pipeline: dataset generation, response collection, scoring, CI gate, flake detection.
- OpenTelemetry tracing with Jaeger UI integration.
- Playwright Edge smoke tests.

## [5.0.0] - 2026-03-10

### Features
- Initial public release.
- VS Code-first Playwright Edge MCP toolkit.
- EdgeSession with 20+ browser actions and verification.
- Natural language command parser.
- Python autonomous agent with OpenAI/Azure model support.
- MCP server with owner lock and persistent Edge profile.
