# Security Best Practices Report

## Executive summary
This codebase is small and understandable, but it currently exposes sensitive data and broad local file-system capabilities through command inputs. If commands can originate from untrusted LLM output or prompt-injected pages (MCP workflows), the current defaults can leak credentials, overwrite arbitrary local files, and persist high-value browser/session artifacts.

## Critical Findings

### 1) SEC-001 - Sensitive data is logged to console and disk
- Severity: Critical
- Location: edge-session.js:83, edge-session.js:130, edge-session.js:466
- Evidence:
  - `this.log('INFO', 'Action started', { action, params });`
  - `this.log('INFO', 'Action completed', { action, durationMs: ..., result });`
  - `fs.appendFileSync(this.logFile, `${line}\n`, 'utf8');`
- Impact: Values typed into forms (including passwords/OTP/token fields) can be written to `logs/edge-agent.log` and console output, creating credential and PII exposure at rest.
- Fix: Redact sensitive keys/values before logging (`password`, `token`, `secret`, OTP-like fields), and disable verbose param/result logging by default.
- Mitigation: Restrict log retention/permissions and document that logs can contain sensitive session data until redaction is implemented.
- False positive notes: None. The code logs full params/results unconditionally.

### 2) SEC-002 - Unvalidated user-controlled file paths allow arbitrary local file write/read access
- Severity: Critical
- Location: edge-session.js:296, edge-session.js:305, edge-session.js:319, edge-session.js:336
- Evidence:
  - `const savePath = path.resolve(params.savePath || ...)`
  - `await locator.setInputFiles(path.resolve(params.filePath));`
  - `const outputPath = path.resolve(params.path || ...)`
  - `const tracePath = path.resolve(params.path || ...)`
- Impact: If command text is influenced by untrusted prompts/tool calls, an attacker can direct reads/writes outside the project directory (for example `C:\...\sensitive`), leading to data exposure or local file tampering.
- Fix: Restrict read/write operations to an allowlisted workspace subtree; reject absolute paths and path traversal; enforce extension/path policy per action.
- Mitigation: Run with OS-level sandboxing and least-privilege filesystem permissions.
- False positive notes: Risk is lower in fully local/manual CLI use, but high in MCP/agent-integrated workflows.

## High Findings

### 3) SEC-003 - Arbitrary HTML injection into active page context
- Severity: High
- Location: edge-session.js:250, edge-session.js:254, agent-logic.js:24, agent-logic.js:28
- Evidence:
  - `wrapper.innerHTML = html;`
  - `temp.innerHTML = html;`
- Impact: Untrusted `html` payloads can inject active content/event handlers in the current page context, enabling session/data abuse on visited sites.
- Fix: Remove raw HTML insertion or gate it behind explicit trusted mode; sanitize with strict allowlist if this capability is required.
- Mitigation: Disable `add` action for untrusted/agent-driven flows.
- False positive notes: Lower risk if only trusted local operator input is accepted.

### 4) SEC-004 - Persistent profile/session + trace capture enabled by default in MCP mode
- Severity: High
- Location: playwright-edge-mcp.js:29, playwright-edge-mcp.js:30, playwright-edge-mcp.js:36, README.md:59
- Evidence:
  - `--save-session`
  - `--save-trace`
  - `--user-data-dir <persistent dir>`
- Impact: Session cookies, auth state, and page snapshots can persist on disk; this increases blast radius on shared machines and can leak sensitive workflow data.
- Fix: Default to ephemeral profile and opt-in persistence; add a `no-trace` mode; document secure storage/cleanup procedures.
- Mitigation: Protect output/profile directories with strict permissions and periodic cleanup.
- False positive notes: If persistent session is required for product UX, keep it but add explicit security controls.

## Medium Findings

### 5) SEC-005 - Global retry can repeat non-idempotent side-effect actions
- Severity: Medium
- Location: edge-session.js:138
- Evidence:
  - `withActionRetry` wraps all actions and retries on any thrown error.
- Impact: Retries can duplicate clicks/submits/messages/download triggers, causing repeated external side effects and potential abuse.
- Fix: Retry only idempotent actions (`goto`, `exists`, `getText`, `waitFor`) or require explicit action-level retry policy.
- Mitigation: Add two-step confirmation for high-impact actions before execution.
- False positive notes: Impact depends on target workflow (higher for message send/payment/form-submit flows).

## Low Findings

### 6) SEC-006 - Validation script does not include all shipped JS entry points
- Severity: Low
- Location: package.json:10
- Evidence:
  - `check` excludes `agent-logic.js` and `playwright-edge-agent.js`.
- Impact: Security or syntax regressions in those files can bypass routine validation.
- Fix: Include all first-party JS files in the `check` command.
- Mitigation: Add CI lint/test coverage for all entry points.
- False positive notes: These files may be legacy, but they are still present and importable.
