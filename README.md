# Agent Live Web

Agent Live Web is a local-first developer agent for repo, terminal, editor, and browser workflows. The main product is the Python agent. Playwright Edge MCP and the Node browser runtime are underlying tool infrastructure used for browser automation, verification, and VS Code integration.

## Product Modes
- Python Agent mode: primary interface for planning, tool use, verification, and hybrid repo plus browser workflows.
- VS Code MCP mode: browser and editor-facing infrastructure for VS Code workflows and direct MCP usage.
- Local browser CLI mode: lower-level browser runtime for operator/debug flows.

## Why This Helps Developers
- Understand repo and browser context in one local workflow.
- Execute tool actions with preview, confirmation, and verification controls.
- Debug failures faster with trace, probe, audit, and live integration tooling.
- Measure stability with repeatable local unit and live MCP test pipelines.

## Local-First Security Model
- Browser profile and MCP runtime stay local by default.
- Sensitive files (`.env`, local profiles, runtime outputs) are excluded from git.
- Side-effect actions should require explicit user confirmation.
- Governance/instruction files are protected by policy unless explicitly requested.
- Python agent command execution uses restricted mode by default (`AGENT_RUN_COMMAND_SECURITY_MODE=restricted`).
- Memory auto-log is off by default (`AGENT_MEMORY_AUTO_LOG=false`) and memory entries are redacted.

## Quickstart
```bash
npm install
npm run install:edge
npm run verify
```

## First-Time Python Agent Setup
1. Copy `.env.example` to `.env`
2. Fill the provider block you actually use
3. Optional quick readiness check:

```bash
npm run agent:preflight
npm run agent:mcp-status
```

4. Start the agent:

```bash
npm run agent:python
```

Session note:
- auto-resume stays locked until one normal assistant turn completes successfully
- immediate quit, setup-only runs, and timeout-only runs do not create resumable session state
- `/save` only persists resumable history after that trusted first turn
- `/mcp` shows the MCP proxy readiness/trust summary when browser tools are connected
- `npm run agent:mcp-status` prints the same MCP proxy report without starting chat
- `npm run agent:mcp-status:json` prints the MCP proxy report as JSON for CI or scripts

## Official Entry Points
- Recommended: Python Agent mode
```bash
npm run agent:preflight
npm run agent:mcp-status
npm run agent:python
```

- VS Code MCP mode
  - Open this workspace in VS Code.
  - Start or restart `playwright-edge` from the MCP panel.
  - Terminal equivalent:
```bash
npm run agent:vscode
```

- Local browser CLI mode
```bash
npm run agent:live-web
```

## First Workflow
Use Python Agent mode first unless you specifically need direct MCP usage in VS Code. The intended product path is:
1. Start the Python agent
2. Let it inspect repo/workspace context
3. Use browser/MCP tools only when the task needs browser state or web workflow verification

If this is a first run:
4. complete one normal turn before expecting session auto-resume on the next startup

## Python Agent Model Config
The Python agent supports provider/model switching without code edits.

- OpenAI/Codex style:
```bash
AGENT_PROVIDER=openai
OPENAI_API_KEY=...
AGENT_MODEL=codex-5.3
```

- Azure style:
```bash
AGENT_PROVIDER=azure
azure_key=...
azure_endpoint_uri=...
azure_deployment_name=...
```

Notes:
- `AGENT_PROVIDER=auto` will prefer OpenAI when `OPENAI_API_KEY` is present, otherwise Azure.
- You can use any future model name by changing only `AGENT_MODEL`.
- Start from `.env.example` instead of creating your env file from scratch.

## OAuth Support (Python Agent Tools)
For OAuth-protected APIs/sites, configure profile and fetch token through tools:

1. `oauth_set_profile` with `profile_name`, `token_url`, `client_id`, `client_secret` (+ optional `scope`/`audience`)
2. `oauth_get_token` with `profile_name`
3. `web_fetch` with `oauth_profile`

Example `web_fetch` auth fields:
- `oauth_profile`: profile name to auto-resolve bearer token
- `auth`: `{ "type": "oauth_profile", "profile_name": "...", "force_refresh": false }`
- `bearer_token`: direct bearer token (if you do not want profile-based flow)

## Maintainability Workflow (Python Agent)
- Run `agent_health_report` after adding or changing tools.
- In interactive Python chat, run `/doctor` to execute the same health report quickly.
- In interactive Python chat, run `/mcp` for the current MCP proxy readiness/trust report.
- Use `agent/agent/ARCHITECTURE_PLAN.md` as the refactor roadmap.
- Use `docs/VSCODE_PYTHON_AGENT_INTEGRATION_PLAN.md` as the plan for adopting VS Code, Pylance, Playwright, and later GitHub-style capability surfaces into the Python agent.
- Use `docs/SAFETY_GATING_DESIGN.md` as the source of truth for confirmation policy, preview flow, and risk classes.
- Keep tool schemas and callable registrations aligned (health report checks this automatically).
- v5.2 MCP live integration roadmap: `docs/V5_2_MCP_INTEGRATION_PLAN.md`.

### Recommended Verification Commands
```bash
npm run verify
npm run agent:test:py:integration
```

### Python Agent Tests
Run targeted unit tests for refactored Python agent modules:

```bash
python -m unittest discover -s agent/agent/tests -p "test_*.py" -v
```

or via npm script:

```bash
npm run agent:test:py
```

GitHub Actions runs this automatically for `agent/**` changes using:
- `.github/workflows/python-agent-tests.yml`

Optional live MCP integration tests (opt-in):

```bash
npm run agent:test:py:integration
```

Set `RUN_MCP_LIVE_TESTS=1` before running integration tests to execute real MCP session checks.

## Tracing and Health
```bash
npm run trace:stack:start
npm run trace:check
npm run trace:triage
npm run health:snapshot
npm run health:snapshot:quiet
```

`npm run health:snapshot` runs JS syntax checks plus Python agent unit tests and writes a timestamped report to `logs/health/`.
Use `npm run health:snapshot:quiet` for CI-friendly minimal console output while preserving full report details.
Report retention is automatic: by default the latest `20` reports are kept in `logs/health/`. Override with `HEALTH_SNAPSHOT_KEEP_LATEST`.

Useful endpoints:
- OTLP: `http://localhost:4318/v1/traces`
- Jaeger UI: `http://localhost:16686`

## Evaluation
```bash
npm run eval:all
npm run eval:gate:normal
```

Release-level checks:
```bash
npm run eval:ci:strict
```

## Public Repo Metadata (Suggested)
- Description: `Local-first developer agent for repo, terminal, editor, and browser workflows with MCP-backed automation, safety gating, and verification.`
- Topics: `playwright, mcp, vscode, edge, browser-automation, web-automation, ai-agent, developer-agent, local-first, verification, safety, nodejs, opentelemetry, jaeger, tracing, observability`
- Current stable release line: `v5.1.x`
- Suggested release title: `v5.1.0 - Local-first agent core + MCP safety`

## Documentation
- Governance guide: `.github/README.md`
- Contributor guide: `CONTRIBUTING.md`
- Global repo instructions: `.github/copilot-instructions.md`
- Website skill: `.github/skills/web-works/SKILL.md`
- Security policy: `SECURITY.md`
- VS Code quickstart: `USAGE_QUICKSTART.md`
- Modes and use cases: `docs/MODES.md`
- Architecture overview: `docs/ARCHITECTURE.md`
- VS Code and Python integration plan: `docs/VSCODE_PYTHON_AGENT_INTEGRATION_PLAN.md`
- Status and roadmap: `docs/ROADMAP.md`
- Troubleshooting: `docs/TROUBLESHOOTING.md`
- Safety gating design: `docs/SAFETY_GATING_DESIGN.md`
- Release verification: `RELEASE_CHECKLIST.md`
- Code ownership policy: `.github/CODEOWNERS`
