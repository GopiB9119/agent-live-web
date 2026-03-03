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

## Dynamic UI reliability constraints
- Before click/type actions in automation flows, enforce preconditions: no active blocker overlay, no unresolved onboarding gate, target is visible and enabled.
- Prefer visible role-based targeting scoped to active containers (`main`, active dialog, chat panel) over global selectors.
- For chatbot turns, require two success signals in logic and tests: user-send confirmed and a new assistant response confirmed.
- Extract chatbot output from the latest assistant message node only; avoid broad DOM text scraping for result evaluation.
- On blocked interactions, follow one retry ladder only: `Escape` → neutral outside click → explicit close control → re-verify → retry with tighter scope.

## Validation after edits
- `node --check playwright-edge-mcp.js`
- For parser/session changes:
  - `node --check edge-session.js`
  - `node --check nl-command-parser.js`
  - `node --check cli-agent.js`
