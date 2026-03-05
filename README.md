# Agent Live Web

This application helps developers automate real-time website workflows in VS Code using Playwright Edge MCP with local-first security, tracing, and evaluation.

## Why This Helps Developers
- Run browser tasks through a single MCP Edge owner flow (`vscode` by default).
- Automate real website actions with step verification and retry controls.
- Debug failures faster with trace + triage scripts.
- Measure stability with repeatable local evaluation pipelines.

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
npm run check
```

## Run
- VS Code MCP mode: start/restart `playwright-edge` from the MCP panel.
- Terminal mode:
```bash
npm run mcp:edge
```

## Python Agent Model Config
The Python agent now supports provider/model switching without code edits.

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
- Use `agent/agent/ARCHITECTURE_PLAN.md` as the refactor roadmap.
- Keep tool schemas and callable registrations aligned (health report checks this automatically).

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
- Description: `VS Code-first Playwright Edge MCP toolkit for reliable real-time web automation, tracing, and evaluation.`
- Topics: `playwright, mcp, vscode, github-copilot, edge, browser-automation, web-automation, ai-agent, nodejs, opentelemetry, jaeger, tracing, observability, qa, automation`
- Current stable release line: `v5.1.x`
- Suggested release title: `v5.1.0 - VS Code Edge MCP stable`

## Documentation
- Governance guide: `.github/README.md`
- Global repo instructions: `.github/copilot-instructions.md`
- Website skill: `.github/skills/web-works/SKILL.md`
- Security policy: `SECURITY.md`
