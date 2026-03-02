---
applyTo: "playwright-edge-mcp.js,edge-session.js,cli-agent.js,nl-command-parser.js,scripts/**/*.js,.vscode/mcp.json"
---

# Playwright Edge Runtime Instructions

## Goal
Keep browser automation fast, deterministic, and owner-safe.

## Rules
- Preserve owner lock behavior (`PLAYWRIGHT_MCP_OWNER`, owner file, lock file).
- Keep `vscode` as the default owner for this repository.
- Do not remove single-owner enforcement unless explicitly asked.
- Keep MCP stdout protocol clean; write diagnostics to stderr.
- Prefer configurable env vars for timeouts and runtime toggles.
- Keep defaults performance-oriented but stable.

## Safety constraints
- Avoid broad command-line relaxations that reduce security.
- Keep blocked origins support in place unless user explicitly asks to disable.
- Do not add automatic destructive process-kill logic.

## Validation after edits
- `node --check playwright-edge-mcp.js`
- For parser/session changes:
  - `node --check edge-session.js`
  - `node --check nl-command-parser.js`
  - `node --check cli-agent.js`
