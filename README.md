# Agent Live Web (VS Code Default)

## Quickstart
```bash
cd /workspaces/agent-live-web
npm install
npm run install:edge
npm run check
```

## Run
- VS Code MCP: restart server `playwright-edge` from the MCP panel.
- Terminal mode:
```bash
npm run mcp:edge
```

Server config in `.vscode/mcp.json` is pinned to `PLAYWRIGHT_MCP_OWNER=vscode` with persistent local profile and no external CDP endpoint.

## What this is for (simple)
- This project controls Edge browser tasks through MCP.
- Tracing helps you see if the app is healthy, slow, or failing.
- This is used for both daily debugging and issue investigation (not only testing).

If you are not sure what to run, use this:
```bash
npm run trace:triage
```
It runs the main checks and shows recent status.

## Tracing
- OTLP endpoint: `http://localhost:4318/v1/traces`
- Jaeger UI: `http://localhost:16686`
- Scope: MCP launcher + browser session runtime
- Excluded: `/workspaces/agent-live-web/agent`

First-time flow:
```bash
npm run trace:stack:start
npm run trace:check
npm run trace:triage
```

Core commands:
```bash
# health
npm run trace:check

# start / stop full stack (collector + Jaeger)
npm run trace:stack:start
npm run trace:stack:stop

# queries
npm run trace:latest
npm run trace:latest:errors
npm run trace:incident
npm run trace:triage
```

Advanced filters:
```bash
TRACE_MIN_DURATION_MS=2000 npm run trace:latest
TRACE_LOOKBACK=24h TRACE_STATUS=ERROR npm run trace:latest
```

## Privacy
- Defaults: file logging off, console telemetry logging off, operator mode on, workspace-restricted outputs.

Enable logs intentionally:
```bash
export EDGE_LOG_TO_CONSOLE=true
export EDGE_WRITE_LOG_FILE=true
npm run agent:live-web
```

Optional overrides:
```bash
export EDGE_DOM_FALLBACK_ON_FAILURE=false
export EDGE_ALLOW_DOM_HTML_ADD=true
export EDGE_RESTRICT_WRITE_TO_WORKSPACE=false
```

## Validation
```bash
npm run check
```

## Evaluation (Local)
Run local evaluation for reliability + performance:

```bash
# optional but recommended before collection
npm run trace:stack:start

# generate dataset, collect real command responses, score report
npm run eval:all
```

Stress profile (stricter, more checks):

```bash
npm run eval:all:stress
```

Which one to use:
- `eval:all` → normal daily health check
- `eval:all:stress` → deeper confidence check before release/change

Advanced phases (run one by one):

```bash
# 1) CI gate + trend history
npm run eval:all:stress
npm run eval:gate

# 2) Flake control (runs multiple times, fails if unstable)
npm run eval:flake

# 3) Failure injection (intentionally creates a failing scenario)
npm run eval:all:failure

# 4) Full CI pipeline in one command
npm run eval:ci
```

What each advanced command does:
- `eval:history` → appends current run to `eval/history.jsonl` and writes `eval/trend.json`
- `eval:gate` → fails when score/failed checks/regressions violate threshold
- `eval:flake` → repeats evaluation and fails on unstable variance
- `eval:all:failure` → verifies the gate can catch real failures

Gate strictness presets:

```bash
# lenient
npm run eval:gate:lenient

# normal (recommended default)
npm run eval:gate:normal

# strict (release-level)
npm run eval:gate:strict
```

Recommended flow by situation:
- Daily: `npm run eval:all` then `npm run eval:gate:normal`
- Before release: `npm run eval:ci:strict`
- Debugging noisy env: `npm run eval:gate:lenient`

Outputs:
- `eval/queries.json`
- `eval/responses.json`
- `eval/report.json`
- `eval/report.md`
- `eval/history.jsonl`
- `eval/trend.json`
- `eval/flake-report.json`
- `eval/flake-report.md`
