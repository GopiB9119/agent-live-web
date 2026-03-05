# v5.2 Target: MCP Integration Hardening

## Objective
Move beyond unit-only coverage and validate real MCP browser automation behavior end-to-end.

## Focus Areas
1. Session reconnect reliability
- Start MCP session, shutdown, reconnect in the same run.
- Verify tool wrappers are re-registered and callable after reconnect.

2. Tab ownership discipline
- Validate blank-tab cleanup and working-tab selection before actions.
- Confirm navigation/actions execute against the owned tab.

3. Retry flow correctness
- Verify retryable tools perform exactly one retry after initial failure.
- Capture attempt count + verification reason in response payload.

## Test Strategy
- Keep unit tests as fast gate (`agent/agent/tests/test_*.py`).
- Add opt-in live integration tests:
  - `agent/agent/tests/integration/test_mcp_live_integration.py`
- Guard live tests with `RUN_MCP_LIVE_TESTS=1` to avoid CI flakiness.

## Execution Plan
1. Stabilize local live test runs on Windows + Edge profile ownership.
2. Add trace artifact capture for integration failures.
3. Introduce a scheduled/nightly workflow for live integration tests.
4. Promote stable live checks into release readiness checklist.
