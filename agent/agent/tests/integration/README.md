# MCP Live Integration Tests

These tests target real MCP session behavior (not only unit-level mocks):
- session reconnect lifecycle across multiple reconnect cycles
- Python manager path registration/execution for `agent_proxy_status`
- tab ownership + blank-tab cleanup behavior
- retry behavior for retryable browser tools
- non-retryable path verification for single-attempt tools
- direct launcher proxy schema augmentation for `confirm` / `confirm_token`
- direct proxy gating for risky browser tool calls
- direct proxy verification/evidence for navigation and screenshot artifact flows

## Prerequisites
- Node dependencies installed (`npm ci`)
- Python dependencies installed (`pip install -r agent/agent/requirements.txt`)
- Playwright Edge MCP launcher present at repo root (`playwright-edge-mcp.js`)
- No competing MCP owner lock from another active session
- By default the live tests create isolated runtime directories under `.playwright-mcp/live-tests/` so they do not reuse the main local profile or owner lock
- By default the live tests also set `PLAYWRIGHT_MCP_HEADLESS=true` unless you explicitly override it
- Optional for artifact-heavy debugging:
  - `PLAYWRIGHT_MCP_SAVE_TRACE=true`
  - `PLAYWRIGHT_MCP_SAVE_SESSION=true`
  - `PLAYWRIGHT_MCP_OUTPUT_DIR=<workspace>/.playwright-mcp/output`
  - `PLAYWRIGHT_MCP_LIVE_KEEP_RUNTIME=1`

## Timeout controls
- `PLAYWRIGHT_MCP_LIVE_INIT_TIMEOUT_SEC` for launcher/session startup
- `PLAYWRIGHT_MCP_LIVE_CALL_TIMEOUT_SEC` for live `tools/list` and `tools/call`
- `PLAYWRIGHT_MCP_LIVE_SHUTDOWN_TIMEOUT_SEC` for teardown/cleanup
- `AGENT_MCP_CONNECT_TIMEOUT_SEC`, `AGENT_MCP_TOOL_TIMEOUT_SEC`, and `AGENT_MCP_SHUTDOWN_TIMEOUT_SEC` for the Python MCP manager itself

## Run (Windows PowerShell)
```powershell
$env:RUN_MCP_LIVE_TESTS='1'
python -m unittest discover -s agent/agent/tests/integration -p "test_*.py" -v
```

## Run (bash)
```bash
RUN_MCP_LIVE_TESTS=1 python -m unittest discover -s agent/agent/tests/integration -p "test_*.py" -v
```

If `RUN_MCP_LIVE_TESTS` is not set to `1`, these tests are skipped. If you explicitly set `PLAYWRIGHT_MCP_OWNER_FILE`, `PLAYWRIGHT_MCP_USER_DATA_DIR`, or `PLAYWRIGHT_MCP_OUTPUT_DIR`, those values override the default isolated live-test paths.

## CI
- GitHub Actions workflow: `.github/workflows/mcp-live-integration.yml`
- Runs on Windows for `main`, pull requests touching MCP/browser/agent paths, schedule, and manual dispatch
- Keeps each test run isolated under `.playwright-mcp/live-tests/`
- Uploads per-run output artifacts and owner-lock files from `.playwright-mcp/live-tests/**`
- On live-test failure, runs `npm run agent:mcp-status:json` first and saves `.playwright-mcp/live-tests/mcp-proxy-status.json`
- On live-test failure, runs the raw MCP probe automatically to capture launcher-side startup evidence

## Raw Probe
If live startup hangs before the suite reports a clean failure, run the raw stdio probe:

```powershell
npm run mcp:probe
npm run mcp:probe -- --profile=minimal --step=initialize
npm run mcp:probe -- --target=playwright-direct --caps=none --init-page=off --shared-context=off --persist-profile=off
```

That script bypasses the Python MCP client, sends raw `initialize` and `tools/list` JSON-RPC frames to `playwright-edge-mcp.js`, and prints the launcher stderr tail plus the isolated runtime files when startup stalls.
Use `--profile=minimal` first when you need to rule out repo-specific startup additions like the init-page script, shared browser context, persistent profile, or default `vision,pdf` caps.
