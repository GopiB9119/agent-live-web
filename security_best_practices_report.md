# Security Best Practices Report

## Executive summary
No critical remote-code-execution or command-injection finding was identified in the current Playwright/VS Code control path. The main remaining security tradeoff is session persistence on disk for convenience (`PLAYWRIGHT_MCP_PERSIST_PROFILE=true`).

## Current findings

### 1) SEC-101 - Persistent browser profile stores authenticated session state on disk
- Severity: High
- Location: `.vscode/mcp.json:9`, `.vscode/mcp.json:20`
- Evidence:
  - `"PLAYWRIGHT_MCP_PERSIST_PROFILE": "true"`
  - `"PLAYWRIGHT_MCP_USER_DATA_DIR": "C:\\Users\\banot\\AppData\\Local\\PlaywrightMCP\\edge-profile"`
- Impact: Cookies/login/session state remain on disk, increasing exposure on shared machines or compromised local accounts.
- Fix: Set `PLAYWRIGHT_MCP_PERSIST_PROFILE` to `"false"` for strict privacy mode.
- Mitigation: Keep device account protected, avoid shared OS accounts, and periodically clear profile data.

## Fixed in this pass

### A) SEC-FIX-201 - File log persistence disabled by default
- Location: `edge-session.js:92-95`, `edge-session.js:112-114`, `edge-session.js:588-594`
- Change: `EDGE_WRITE_LOG_FILE` now defaults to `false`; log file is only written when explicitly enabled.

### B) SEC-FIX-202 - Console telemetry disabled by default
- Location: `edge-session.js:96-99`, `edge-session.js:585-587`
- Change: `EDGE_LOG_TO_CONSOLE` now defaults to `false`; console logs are opt-in.

### C) SEC-FIX-203 - Raw DOM HTML injection action blocked by default
- Location: `edge-session.js:87-90`, `edge-session.js:185-187`
- Change: `add` action now requires explicit `EDGE_ALLOW_DOM_HTML_ADD=true`.

### D) SEC-FIX-204 - Output paths constrained to workspace by default
- Location: `edge-session.js:80-86`, `edge-session.js:395`, `edge-session.js:418`, `edge-session.js:435`, `edge-session.js:442-456`
- Change: download/screenshot/trace writes are blocked outside workspace unless explicitly disabled.

### E) SEC-FIX-205 - Ownership conflict validation restored
- Location: `playwright-edge-mcp.js:129-133`, `playwright-edge-mcp.js:135-141`
- Change: active owner is validated before writing owner file, preventing silent owner override.

### F) SEC-FIX-206 - VS Code ownership hardened
- Location: `scripts/set-mcp-owner.js:5`, `scripts/set-mcp-owner.js:8`
- Change: owner helper now accepts only `vscode|none` (removed `python` option).
