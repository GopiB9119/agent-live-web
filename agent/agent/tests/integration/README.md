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
