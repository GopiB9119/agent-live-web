# Contributing

This project is a local-first developer agent with Python as the main product surface and Playwright Edge MCP as browser/tooling infrastructure.

## Before You Change Anything
- Read `README.md`
- Read `docs/MODES.md`
- Read `docs/ARCHITECTURE.md`
- Read `docs/ROADMAP.md`
- Read `docs/SAFETY_GATING_DESIGN.md`
- For Python agent work, read `agent/agent/ARCHITECTURE_PLAN.md`

## Local Setup
```bash
npm install
python -m pip install -r agent/agent/requirements.txt
npm run install:edge
```

For Python agent work:
- copy `.env.example` to `.env`
- fill only the provider block you use

## Primary Entry Points
- Python agent: `npm run agent:python`
- VS Code MCP mode: `npm run agent:vscode`
- Local browser CLI mode: `npm run agent:live-web`

## Required Verification
Run this before opening a PR:
```bash
npm run verify
```

For browser/MCP runtime changes, also run:
```bash
npm run agent:test:py:integration
```

Set `RUN_MCP_LIVE_TESTS=1` for the live integration suite.

## Change Rules
- Keep Python agent behavior and MCP/browser behavior consistent with the shared safety model.
- Do not bypass preview/confirm/blocked policy paths for side-effecting tools.
- Prefer focused module changes over growing monolith files.
- Do not change `.github/**` or `AGENTS.md` unless the change is explicitly about governance/instructions.

## Pull Requests
- Keep the scope narrow and explain the user-visible effect.
- Include verification performed and exact commands used.
- Call out residual risks, skipped tests, or environment-specific limitations.

## High-Risk Areas
- `agent/agent/tools.py` and tool registry wiring
- `agent/agent/mcp_tools.py`
- `playwright-edge-mcp.js`
- `edge-session.js`
- safety and confirmation logic under `agent/agent/architecture/`

## When In Doubt
- Default to Python-first product behavior.
- Treat browser automation as a controlled tool surface, not the primary product identity.
