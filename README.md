# Agent Live Web

This application helps developers automate real-time website workflows in VS Code using Playwright Edge MCP with local-first security, tracing, and evaluation.

## Why This Helps Developers
- Run browser tasks through a single MCP Edge owner flow (`vscode` by default).
- Automate real website actions with step verification and retry controls.
- Debug failures faster with trace + triage scripts.
- Measure stability with repeatable local evaluation pipelines.

## How It Should Work
The intended developer workflow is:
- the agent understands the developer's codebase before making changes
- the developer provides work intent in a reusable skill or task format
- the agent understands what is happening on the target website through Playwright
- the agent chooses the best path: without API, with API, or hybrid
- the agent completes the work, writes reusable scripts, and verifies those scripts
- the agent keeps its response grounded in verified tool results instead of guessing beyond what was actually observed
- the agent remembers what was already completed and what the next steps are through saved sanitized session state and execution artifacts
- the agent gives developers a brief explanation of what happened, what was changed, and what to do next

Two main execution paths matter:

### Without API
- use VS Code agent flow by default
- inspect the website deeply through Playwright
- execute the workflow from the user-provided skill
- generate reusable browser automation scripts
- test the script and return a short developer-facing summary

### With API
- still understand the website first through Playwright so the real flow is clear
- then use APIs where they make the work faster and more deterministic
- generate reusable scripts that combine API work with browser verification when needed
- test the script and return a short developer-facing summary

The product should help developers move faster by combining codebase understanding, website understanding, reusable skills, script generation, and verification in one workflow.

Workflow outputs should also leave behind a sanitized reusable execution artifact with a brief developer-facing summary, so a successful run is easier to reuse as a script, skill, or follow-up task.

## Local-First Security Model
- Browser profile and MCP runtime stay local by default.
- Sensitive files (`.env`, local profiles, runtime outputs) are excluded from git.
- Keys, cookies, auth headers, secret-bearing URLs, and sensitive runtime outputs should be redacted before logs, traces, saved artifacts, or model-visible summaries.
- DOM mutation actions like `add` and `delete` are disabled by default and require explicit opt-in (`EDGE_ALLOW_DOM_HTML_ADD=true`, `EDGE_ALLOW_DOM_DELETE=true`).
- Governance/instruction files are protected by policy unless explicitly requested.
- Python agent command execution uses restricted mode by default (`AGENT_RUN_COMMAND_SECURITY_MODE=restricted`).
- Memory auto-log is off by default (`AGENT_MEMORY_AUTO_LOG=false`) and memory entries are redacted.

## Quickstart
```bash
npm install
npm run install:edge
cp .env.example .env     # then edit .env with your API keys
npm test                 # syntax check + 24 JS unit tests
```

See `docs/USAGE_QUICKSTART.md` for the full zero-to-working guide covering all three runtimes.

## Run
| Runtime | Command | Best for |
|---------|---------|----------|
| VS Code MCP | Start `playwright-edge` in MCP panel | Interactive work, debugging |
| Terminal CLI | `npm run agent:live-web` | Supervised browser control |
| Python agent | `python agent/agent/agent.py` | CI, batch, scheduling |

## Test
```bash
npm test              # JS syntax check + JS unit tests (44 tests)
npm run test:py       # Python unit tests (92 tests)
npm run test:all      # Both JS and Python in one command (136 total)
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
- `bearer_token`: direct bearer token for local use when needed, but tool output redacts secret-bearing values and does not echo raw token material back

Notes:
- prefer `oauth_profile` over passing raw tokens in prompts or tool arguments when possible
- `oauth_get_token` keeps raw token output disabled by default and is intended for token acquisition/caching, not secret disclosure

## Maintainability Workflow (Python Agent)
- Run `agent_health_report` after adding or changing tools.
- In interactive Python chat, run `/doctor` to execute the same health report quickly.
- Use `agent/agent/ARCHITECTURE_PLAN.md` as the refactor roadmap.
- Keep tool schemas and callable registrations aligned (health report checks this automatically).
- v5.2 MCP live integration roadmap: `docs/V5_2_MCP_INTEGRATION_PLAN.md`.

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
- Description: `VS Code-first Playwright Edge MCP toolkit for reliable real-time web automation, tracing, and evaluation.`
- Topics: `playwright, mcp, vscode, github-copilot, edge, browser-automation, web-automation, ai-agent, nodejs, opentelemetry, jaeger, tracing, observability, qa, automation`
- Current stable release line: `v6.0.x`
- Suggested release title: `v6.0.0 - VS Code Edge MCP stable`

## Documentation
- Governance guide: `.github/README.md`
- Global repo instructions: `.github/copilot-instructions.md`
- Product architecture: `docs/PRODUCT_ARCHITECTURE.md`
- Trust and reliability plan: `docs/TRUST_RELIABILITY_EXECUTION_PLAN.md`
- Website skill: `.github/skills/web-works/SKILL.md`
- Security policy: `SECURITY.md`
- Release verification: `docs/RELEASE_CHECKLIST.md`
- Security audit report: `docs/security_best_practices_report.md`
- Memory system guide: `docs/MEMORY_USAGE.md`
- API reference (58 tools): `docs/API_REFERENCE.md`
- Changelog: `CHANGELOG.md`
- Code ownership policy: `.github/CODEOWNERS`

## Project Structure
```
agent-live-web/
├── index.js                    # npm entry point (exports EdgeSession, parseCommand, tracing)
├── edge-session.js             # Core: Playwright Edge browser session + actions + verification
├── nl-command-parser.js        # Core: Natural language → browser action parser
├── playwright-edge-mcp.js      # MCP server launcher (owner lock, profile, child process)
├── cli-agent.js                # Interactive REPL for browser automation
├── tracing.js                  # OpenTelemetry SDK integration
│
├── agent/agent/                # Python AI agent layer
│   ├── agent.py                # Conversation loop, model provider, session state
│   ├── config.py               # Environment loading, model client, runtime constants
│   ├── tools.py                # Tool wiring: instantiates managers, builds registry
│   ├── runtime_utils.py        # Shared config, SSRF checks, redaction, grounding
│   ├── SYSTEM_PROMPT.md        # Agent personality and tool routing rules
│   ├── command_tools.py        # Restricted/permissive command execution
│   ├── fs_tools.py             # Workspace-scoped file operations + code analysis
│   ├── web_tools.py            # Web fetch with SSRF protection + OAuth
│   ├── memory_tools.py         # Daily logs, long-term memory, hybrid search
│   ├── mcp_tools.py            # MCP session lifecycle + browser tab ownership
│   ├── oauth_tools.py          # Client credentials / refresh token flows
│   ├── workflow_tools.py       # Multi-step plans, workflow execution, autopilot
│   ├── diagnostics_tools.py    # Tool catalog + agent health report
│   ├── tooling/                # Tool schema registry
│   │   ├── schemas.py          # AGENT_TOOLS: canonical tool definitions
│   │   └── registry.py         # Auto-registration + callable mapping
│   └── tests/                  # Python unit tests (87 tests)
│       ├── test_*.py           # Per-module test files
│       └── integration/        # Live MCP integration tests (opt-in)
│
├── tests/                      # JS unit tests (44 tests)
│   ├── edge-session.test.js    # EdgeSession action + verification tests
│   └── smoke-chain.test.js     # NL parser → session → response chain tests
│
├── scripts/                    # Automation & DevOps scripts
│   ├── eval/                   # Evaluation pipeline (dataset, scoring, CI gate)
│   ├── health-snapshot.js      # Full health report (JS + Python)
│   ├── mcp-preflight.js        # 8 automated MCP runtime checks
│   ├── validate-governance.js  # .github structure validator
│   └── ...                     # Tracing, profiling, and utility scripts
│
├── docs/                       # Documentation
│   ├── PRODUCT_ARCHITECTURE.md
│   ├── USAGE_QUICKSTART.md
│   ├── RELEASE_CHECKLIST.md
│   └── ...                     # Security report, trust plan, integration plan
│
├── eval/                       # Evaluation data (generated, gitignored)
├── .github/                    # Copilot agents, instructions, skills, workflows
├── .vscode/                    # MCP config + editor settings
└── package.json                # npm scripts (50+ commands)
```
