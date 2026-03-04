# Agent Live Web

VS Code-first Playwright Edge MCP toolkit for reliable real-time web automation, tracing, and evaluation.

This project helps developers automate web workflows from VS Code with stronger reliability checks, safer defaults, and local-first runtime behavior.

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

## Tracing and Health
```bash
npm run trace:stack:start
npm run trace:check
npm run trace:triage
```

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
