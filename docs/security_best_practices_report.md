# Security Best Practices Report

## Executive Summary
This review focused on command execution safety, network-fetch SSRF controls, memory/log handling, repository hygiene, and release policy clarity.  
Primary risks were unrestricted shell execution and private-network fetch exposure. Both were hardened in this pass.

## Critical Findings

### SEC-001 - Unrestricted shell execution path from model tool call
- Severity: Critical
- Location: `agent/agent/tools.py:2583-2649` (before fix)
- Impact: Prompt-injected tool usage could execute arbitrary PowerShell commands on host.
- Fix applied:
  - Added `security_mode` with default `restricted`.
  - Added safe-command allowlist for restricted mode.
  - Added explicit dangerous command gate requiring:
    - `allow_dangerous=true`
    - `confirm=true`
    - env `AGENT_ALLOW_DANGEROUS_COMMANDS=1`
- Updated location: `agent/agent/tools.py:2583-2689`.

## High Findings

### SEC-002 - SSRF/private-network fetch exposure
- Severity: High
- Location: `agent/agent/tools.py:2652-2711` (before fix)
- Impact: Tool could fetch localhost/link-local/private hosts and return sensitive internal content.
- Fix applied:
  - Added hostname/IP private/local checks (DNS-resolved too).
  - Blocked private/local fetch by default.
  - Added controlled override via `allow_private_hosts=true` or `AGENT_WEB_FETCH_ALLOW_PRIVATE_HOSTS=1`.
- Updated location: `agent/agent/tools.py:2693-2768`.

### SEC-003 - Memory logging could persist sensitive data
- Severity: High
- Location:
  - `agent/agent/agent.py:30` (auto-log default)
  - `agent/agent/agent.py:255,344` (log calls)
  - `agent/agent/tools.py:2049-2108` (memory tools)
- Impact: User prompts/responses containing secrets could be persisted in markdown memory files.
- Fix applied:
  - Default `AGENT_MEMORY_AUTO_LOG=false`.
  - Added sensitive pattern redaction + truncation in agent-level memory logging.
  - Added redaction in `memory_log` and `memory_promote` tool handlers.

## Medium Findings

### SEC-004 - Terminal auto-approve enabled in VS Code settings
- Severity: Medium
- Location: `.vscode/settings.json:2-3` (before fix)
- Impact: Reduces operator approval barrier for terminal execution.
- Fix applied:
  - Set `chat.tools.terminal.autoApprove.start` to `false`.

### SEC-005 - Hardcoded local user path in Python MCP env
- Severity: Medium
- Location: `agent/agent/tools.py:2802-2803` (before fix)
- Impact: Privacy leak and poor portability.
- Fix applied:
  - Replaced with env/runtime-root derived paths.

## Low Findings

### SEC-006 - Binary cache files tracked in repository
- Severity: Low
- Location: `agent/agent/__pycache__/*.pyc`
- Impact: Unnecessary binary artifacts in source control.
- Fix applied:
  - Added `__pycache__/` and `*.pyc` to `.gitignore`.
  - Recommended untracking tracked pyc files from git index.

### SEC-007 - Security policy file was template content
- Severity: Low
- Location: `SECURITY.md` (before fix)
- Impact: Vulnerability reporting process unclear.
- Fix applied:
  - Rewrote `SECURITY.md` with supported versions, reporting channel, response targets, and disclosure guidance.

## Residual Risks
1. `run_command` still allows execution when explicitly elevated by operator settings.
2. `web_fetch` private host override remains available for trusted internal workflows.
3. Existing tracked memory files and pyc files should be removed from git index if public release is intended.

## Recommended Next Actions
1. Untrack local memory artifacts and pyc files from git history/index.
2. Keep `AGENT_ALLOW_DANGEROUS_COMMANDS` unset in normal environments.
3. Add CI checks to fail if memory artifacts or pyc files are reintroduced.
