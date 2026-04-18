# MCP Live Integration Tests

These tests target real MCP session behavior (not only unit-level mocks):
- session reconnect lifecycle
- tab ownership flow
- retry behavior for retryable browser tools

## Prerequisites
- Node dependencies installed (`npm ci`)
- Python dependencies installed (`pip install -r agent/agent/requirements.txt`)
- Playwright Edge MCP launcher present at repo root (`playwright-edge-mcp.js`)
- No competing MCP owner lock from another active session

## Run (Windows PowerShell)
```powershell
$env:RUN_MCP_LIVE_TESTS='1'
python -m unittest discover -s agent/agent/tests/integration -p "test_*.py" -v
```

## Run (bash)
```bash
RUN_MCP_LIVE_TESTS=1 python -m unittest discover -s agent/agent/tests/integration -p "test_*.py" -v
```

If `RUN_MCP_LIVE_TESTS` is not set to `1`, these tests are skipped.

## Failure artifacts
- On a live-test failure, artifacts are written to `logs/mcp-live/<test-id>/`.
- Captured files include the unittest traceback, sanitized current tab state, sanitized browser accessibility snapshot, and an index of MCP output files.
- Playwright trace saving is enabled by default for live tests through `PLAYWRIGHT_MCP_SAVE_TRACE=true`.
- Successful runs keep MCP output in a temporary directory and do not leave empty artifact folders behind.
- Set `MCP_LIVE_CAPTURE_TRACE=0` if you need to disable trace generation for a local run.
- Raw copied MCP output files are disabled by default to avoid persisting secret-bearing traces or screenshots; set `MCP_LIVE_COPY_RAW_OUTPUT_FILES=1` to opt in when you explicitly need them.
- Set `MCP_LIVE_ARTIFACTS_DIR=/custom/path` to redirect persisted failure artifacts.
