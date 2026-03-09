# Troubleshooting

Use this document when the local developer agent, MCP runtime, or live integration tests fail in ways that are not obvious from the normal quickstart.

## 1. Start with the baseline
Run:

```bash
npm run verify
```

This confirms:
- JS syntax checks
- JS browser/MCP safety unit tests
- Python agent unit tests

If this fails, fix the baseline before debugging live MCP behavior.

## 1a. Python agent will not start because model config is missing
Symptom:
- `npm run agent:python`
- startup says the model client is not configured

Fix:
1. Copy `.env.example` to `.env`
2. Fill either:
   - OpenAI block
   - or Azure block
3. Check readiness:

```bash
npm run agent:preflight
```

4. Retry:

```bash
npm run agent:python
```

If you are unsure which mode to use:
- read `docs/MODES.md`

## 1b. Session auto-resume is still locked
Symptom:
- `npm run agent:preflight` shows:
  - `session_auto_resume_ready: False`
  - `session_resume_reason: complete one successful startup before auto-resume`
- `/status` shows `startup_completed: False`

This is usually expected.

Current rule:
- the Python agent only creates resumable session state after one normal assistant turn completes
- immediate quit does not count
- setup-only or preflight-only runs do not count
- timeout-only runs do not count
- tool-assisted turns only count when the tool results themselves are successful

What to do:
1. Start the Python agent:

```bash
npm run agent:python
```

2. Complete one normal turn that returns a final assistant response.
3. Run `/status` or `npm run agent:preflight` again.

Expected after that:
- `startup_completed: True`
- `startup_completion_reason: direct_answer` or `tool_assisted_answer`
- `startup_used_tools: False` or `True`
- `startup_trust: direct answer` or `tool-assisted verified`
- `session_auto_resume_ready: True`

## 1c. `/save` says checkpoint skipped
Symptom:
- `/save`
- agent prints: `Checkpoint skipped: resume state is locked until first successful turn.`

This is expected on first-run or failed-run sessions.

Reason:
- resumable history is intentionally blocked until the current run becomes trusted

What to do:
1. Complete one normal assistant turn first.
2. Run `/save` again if you want an immediate checkpoint.

## 1d. `/reset` cleared my resumable session state
Symptom:
- `/reset`
- agent reports that resume state was cleared until one successful turn completes

This is expected when the current run is still untrusted.

Reason:
- the agent now prefers clearing stale resume state over carrying forward history from an untrusted startup

If you want resumable state again:
1. complete one normal assistant turn
2. optionally run `/save`

## 2. Python agent starts but MCP tools are unavailable
Symptom:
- `npm run agent:python`
- Agent starts, then prints that MCP connection failed and continues without external MCP tools

Common causes:
- Playwright MCP launcher could not start
- Owner-lock/profile path conflict
- Windows local permission issue on browser/runtime directories
- Browser target not available

Checks:
```bash
npm run agent:mcp-status
npm run agent:mcp-status:json
npm run agent:vscode
npm run mcp:probe -- --target=launcher --profile=minimal --step=tools-list
```

When MCP does connect, `npm run agent:preflight` and `/status` now also show:
- `mcp_proxy_runtime_status`
- `mcp_proxy_startup_trust`
- `mcp_proxy_resume_state`
- `mcp_proxy_summary`
- `/mcp` prints the same MCP proxy status block directly inside Python agent mode

If the probe fails:
- inspect the stderr tail from the probe
- inspect `.playwright-mcp/live-tests/**` or your configured runtime directory
- check owner lock and output artifacts

## 3. VS Code MCP owner lock mismatch
Symptom:
- VS Code MCP panel or launcher reports owner mismatch or active owner conflict

What to do:
1. Stop any other MCP/browser session using the same profile.
2. Restart `playwright-edge` in VS Code or rerun:

```bash
npm run agent:vscode
```

3. If the lock persists, clear the explicit owner marker:

```bash
node scripts/set-mcp-owner.js none
```

4. Retry the MCP startup.

Notes:
- default shared owner is `vscode`
- live tests use isolated runtime directories under `.playwright-mcp/live-tests/` and should not reuse the shared profile
- in direct MCP mode, call `agent_proxy_status` to inspect:
  - `runtime_status`
  - `startup_trust: direct MCP proxy`
  - `resume_state: not applicable in direct MCP mode`
  - owner-lock and safety settings reported by the proxy

## 4. Live MCP integration tests fail
Run:

```bash
RUN_MCP_LIVE_TESTS=1 npm run agent:test:py:integration
```

On Windows PowerShell:

```powershell
$env:RUN_MCP_LIVE_TESTS='1'
npm run agent:test:py:integration
```

Useful env vars:
- `PLAYWRIGHT_MCP_LIVE_RUNTIME_ROOT`
- `PLAYWRIGHT_MCP_LIVE_INIT_TIMEOUT_SEC`
- `PLAYWRIGHT_MCP_LIVE_CALL_TIMEOUT_SEC`
- `PLAYWRIGHT_MCP_HEADLESS`
- `PLAYWRIGHT_MCP_SAVE_TRACE`
- `PLAYWRIGHT_MCP_SAVE_SESSION`
- `PLAYWRIGHT_MCP_LIVE_KEEP_RUNTIME`

When the live suite fails:
1. rerun with trace/session saving enabled
2. keep the runtime directory
3. run the raw probe

Example:

```powershell
$env:RUN_MCP_LIVE_TESTS='1'
$env:PLAYWRIGHT_MCP_SAVE_TRACE='true'
$env:PLAYWRIGHT_MCP_SAVE_SESSION='true'
$env:PLAYWRIGHT_MCP_LIVE_KEEP_RUNTIME='1'
npm run agent:test:py:integration
npm run mcp:probe -- --target=launcher --profile=minimal --step=tools-list
```

## 5. Raw MCP probe usage
Use the probe when startup or `initialize` appears hung.

Examples:

```bash
npm run mcp:probe
npm run mcp:probe -- --profile=minimal --step=initialize
npm run mcp:probe -- --target=launcher --profile=minimal --step=tools-list
npm run mcp:probe -- --target=playwright-direct --caps=none --init-page=off --shared-context=off --persist-profile=off
```

What it helps isolate:
- launcher/proxy issue vs direct Playwright MCP issue
- client transport problem vs child server startup problem
- repo-specific startup additions vs minimal runtime

## 6. `Access is denied` on Windows
Typical symptoms:
- `WinError 5`
- child profile/runtime directory exists but nested writes fail
- temp-directory based tests fail unpredictably

What this repo now assumes:
- do not rely on generic OS temp dirs for Python tests
- use repo-local runtime/test directories instead

If you see this in runtime paths:
- prefer `.playwright-mcp/live-tests/`
- prefer `.agent-state/`
- avoid reusing stale locked profile directories

If you see this in custom local experiments:
- create directories with normal `Path.mkdir()` under the repo
- avoid `tempfile.TemporaryDirectory()` and `tempfile.mkdtemp()` for Windows-local test harnesses here

## 7. Browser actions are blocked or require confirmation
This is usually expected behavior, not a bug.

Read:
- `docs/SAFETY_GATING_DESIGN.md`

Current model:
- read-only actions may run immediately
- scoped writes may require verification
- risky browser/file/command actions may require preview or same-run confirmation
- destructive actions may be blocked by default

Check:
- whether the tool returned `preview_required`
- whether a `confirm_token` was returned
- whether the task was classified as destructive or high-risk

## 8. What to attach when reporting a bug
Include:
- exact command run
- OS, Python, Node versions
- whether you used `RUN_MCP_LIVE_TESTS=1`
- stderr tail or error text
- relevant files under `.playwright-mcp/live-tests/**/output/`
- whether the failure is baseline (`npm run verify`) or live-only

Also use:
- `.github/ISSUE_TEMPLATE/bug_report.md`

## 9. If you are unsure where the failure belongs
- `npm run verify` fails:
  - baseline regression
- `npm run verify` passes but live integration fails:
  - launcher/MCP/browser/runtime issue
- Python agent runs without MCP tools:
  - startup/connectivity/runtime environment issue
- VS Code MCP panel fails but terminal launcher works:
  - likely owner/config/VS Code surface issue
