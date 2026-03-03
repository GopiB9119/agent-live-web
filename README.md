# Agent Live Web (VS Code Default)

This repository is now configured for VS Code Copilot MCP as the default and primary mode.

## Setup
```bash
cd /workspaces/agent-live-web
npm install
npm run install:edge
```

## Default run mode (VS Code)
1. Open this workspace in VS Code.
2. Restart MCP server `playwright-edge` from the MCP panel.

The server config in `.vscode/mcp.json` is already pinned to VS Code owner:
- `PLAYWRIGHT_MCP_OWNER=vscode`
- persistent local Edge profile enabled
- no external CDP endpoint required

## Optional terminal start
```bash
cd /workspaces/agent-live-web
npm run mcp:edge
```

This command auto-claims `vscode` owner and uses persistent profile mode by default.

## Tracing (VS Code Copilot MCP only)
Tracing is enabled for the VS Code MCP server (`playwright-edge`) via `.vscode/mcp.json`.

- Scope: MCP launcher + browser session runtime in this workspace
- Excluded: `/workspaces/agent-live-web/agent` (untouched)
- Default OTLP endpoint: `http://localhost:4318/v1/traces`

If needed, change tracing env values under `.vscode/mcp.json`:
- `EDGE_TRACING_ENABLED`
- `EDGE_TRACING_SERVICE_NAME`
- `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT`

Quick health check before MCP start:
```bash
npm run trace:check
```

Start local collector (required if `trace:check` fails on localhost):
```bash
npm run trace:collector:start
npm run trace:check
```

Start full tracing stack with Jaeger UI:
```bash
npm run trace:stack:start
```

Jaeger UI:
- `http://localhost:16686`
- Service name examples: `agent-live-web-vscode-mcp`, `agent-live-web-cli`

Stop local collector:
```bash
npm run trace:collector:stop
```

Stop full tracing stack:
```bash
npm run trace:stack:stop
```

Show latest runtime MCP traces (excludes manual smoke events):
```bash
npm run trace:latest
```

## Privacy defaults
- CLI file logging is disabled by default.
- CLI console telemetry logging is disabled by default.
- Local operator mode is enabled by default, and normal Playwright actions now auto-fallback to DOM strategy on failure.
- Download/screenshot/trace outputs are restricted to your workspace by default.
- To enable logs intentionally:
```bash
export EDGE_LOG_TO_CONSOLE=true
export EDGE_WRITE_LOG_FILE=true
npm run agent:live-web
```

Optional overrides (only if you explicitly need them):
```bash
export EDGE_DOM_FALLBACK_ON_FAILURE=false
export EDGE_ALLOW_DOM_HTML_ADD=true
export EDGE_RESTRICT_WRITE_TO_WORKSPACE=false
```

## Validation
```bash
npm run check
```
