# Safety Gating Design

## Purpose
Define one shared safety-gating system for `agent-live-web` so Python Agent mode, VS Code Agent mode, and Node/MCP browser infrastructure all follow the same policy model.

This document is the concrete design target for:
- action classification
- confirmation policy
- preview and commit flow
- audit logging
- verification requirements
- tool-by-tool risk mapping

## Current Gaps
- README says side-effect actions should require explicit confirmation, but current Python file tools can write, move, patch, and delete immediately.
- Browser actions are partially protected by workspace and owner locks, but not by one shared confirmation model.
- `run_command` has restricted/permissive logic, but command policy is separate from filesystem and browser policy.
- Memory and OAuth actions are security-sensitive, but they are not yet governed by the same action-class model.

## Goals
- One safety model for all entry surfaces.
- Safe defaults for local developer workflows.
- Low friction for read-only and scoped reversible work.
- Explicit friction for destructive or external side effects.
- Auditability for every state-changing action.
- Deterministic policy decisions that can be tested.

## Non-Goals
- Guaranteeing zero failures.
- Requiring user confirmation for every code edit.
- Replacing existing workspace-path or SSRF checks.

## Core Decision Model
Every tool call should pass through a shared policy engine before execution.

Decision outputs:
- `allow`: execute immediately
- `allow_with_verification`: execute, but verifier receipt is mandatory
- `preview_required`: produce a dry-run preview first
- `confirm_required`: block until explicit same-run confirmation is supplied
- `blocked`: do not execute

Every decision should return:
- `tool_name`
- `action_class`
- `risk_level`
- `decision`
- `reason_codes`
- `requires_verification`
- `confirm_token` when confirmation is required
- `preview_summary` when preview is possible

## Action Classes
### A0 `read_only`
No state change. Reads repo, browser, diagnostics, or logs.

Examples:
- file reads
- search
- codebase analysis
- browser snapshot
- browser tab listing
- health and diagnostics

Default decision:
- `allow`

### A1 `scoped_reversible_write`
Small local change, bounded to explicit files or browser fields, with clear rollback path.

Examples:
- patch lines in one file
- insert lines in one file
- fill a search box
- navigate browser to a page

Default decision:
- `allow_with_verification`

Escalates to `preview_required` if:
- target scope is ambiguous
- many files are affected
- file path was inferred instead of explicit
- operation touches sensitive paths

### A2 `broad_local_write`
Local state change that is larger, less reversible, or could affect multiple files or browser state broadly.

Examples:
- create new files
- overwrite a file
- copy or move files
- run permissive local build or write commands
- browser file upload

Default decision:
- `preview_required`

Escalates to `confirm_required` if:
- overwrite is requested
- operation affects more than one top-level area
- path includes secrets, config, workflows, or governance files
- command writes outside clearly requested scope

### A3 `external_side_effect`
State change outside the local workspace or actions that can affect remote systems, accounts, or external data.

Examples:
- browser submit/send/purchase flows
- authenticated API writes
- pushing git
- downloading remote files into workspace
- retrieving OAuth tokens for real services

Default decision:
- `confirm_required`

### A4 `destructive`
Delete, destroy, or hard-to-recover operations.

Examples:
- file deletion
- recursive delete
- browser delete/remove actions
- DOM mutation helpers
- dangerous shell commands

Default decision:
- `blocked`

Allowed only in explicit operator mode with:
- same-run confirmation
- action fingerprint match
- policy allow flag
- audit record

## Risk Escalation Factors
The policy engine should compute a base action class, then escalate using these factors:

- `scope_size`
  - one file / one tab / one field
  - many files / recursive / project-wide
- `path_sensitivity`
  - `.env`
  - secrets
  - auth config
  - CI workflows
  - release or governance docs
- `network_target`
  - local/private
  - remote public
  - authenticated remote
- `destructive_intent`
  - delete
  - overwrite
  - recursive
  - DOM removal
- `ambiguity`
  - inferred target
  - fuzzy matched tool
  - broad command text
- `externality`
  - remote side effect
  - account state
  - purchase/send/submit/push
- `verification_strength`
  - can result be verified deterministically?

## Confirmation Model
Confirmation must be same-run and action-bound.

Required fields:
- `confirm=true`
- `confirm_token=<issued token>`

Token rules:
- generated from a hash of `tool_name + normalized_arguments + decision timestamp`
- expires after a short TTL such as 10 minutes
- invalidated if arguments change

This prevents a prompt or tool from reusing approval for a different action.

## Preview Model
When possible, state-changing tools should support dry-run previews.

Required preview behavior:
- show target path or target page element
- show affected scope
- show expected change summary
- show risk classification
- show verification plan

Examples:
- `fs_patch`: return diff stats and changed regions without writing
- `fs_write`: return create vs overwrite summary
- `run_command`: return classification and reason without executing
- browser submit-like actions: return intended target and evidence needed after commit

## Verification Model
Any action above `A0` requires a verifier receipt.

Verifier receipt should include:
- `expected_result`
- `observed_result`
- `verification_status`
- `confidence`
- `evidence`
- `next_safe_step`

Examples:
- file edit: before/after hash, diff size, optional tests
- command run: exit code, stdout/stderr summary, impacted files if known
- browser action: URL/title/DOM evidence, tab state, screenshot or snapshot hash when useful

## Audit Logging
Every `A1+` action should emit a structured audit event to a local log such as:
- `.agent-state/safety-events.jsonl`

Audit fields:
- timestamp
- user mode (`python` or `vscode`)
- tool name
- normalized args summary
- action class
- policy decision
- confirm token id if issued
- execution status
- verification status
- redacted evidence summary

Do not log raw secrets, bearer tokens, OAuth secrets, or full sensitive content.

## Concrete Tool Mapping
### Python filesystem tools
`fs_list`, `fs_read`, `fs_read_batch`, `fs_search`, `fs_analyze_file`, `codebase_analyze`
- Class: `A0`
- Default: `allow`

`fs_edit_lines`, `fs_insert_lines`, `fs_patch`
- `dry_run=true`: `A0`
- bounded single-file write: `A1`
- broad or ambiguous edit: `A2`
- Default: `allow_with_verification` for explicit scoped edits, otherwise `preview_required`

`fs_write`
- create new non-sensitive file: `A2`
- overwrite existing file: escalate to `A3`-style confirmation semantics locally
- Default: `preview_required`
- Confirm if overwrite or sensitive path

`fs_copy`, `fs_move`
- small explicit path: `A2`
- overwrite or cross-area move: `confirm_required`

`fs_delete`
- file delete: `A4`
- recursive delete: `A4`
- Default: `blocked` unless operator mode + confirmation

### Python command tools
`run_command`
- restricted safe read-only commands: `A0`
- safe local write commands in permissive mode: `A2`
- dangerous commands: `A4`
- Default: existing restricted mode remains, but policy engine should classify before manager execution

### Python web and OAuth tools
`web_fetch`
- anonymous read to allowed public host: `A0`
- authenticated fetch: `A2`
- remote write APIs later: `A3`

`oauth_set_profile`
- secret-bearing local configuration write: `A2`
- confirm if writing persistent credentials

`oauth_get_token`
- authenticated remote token retrieval: `A3`
- `confirm_required`

### Memory tools
`memory_get`, `memory_search`, `memory_bootstrap`, `memory_reindex`
- reads or local maintenance: `A0` or `A1`

`memory_log`, `memory_promote`
- privacy-sensitive persistent write: `A2`
- only auto-allow when user explicitly asked to remember or project policy enables it

### Browser/MCP tools
Read-only or observational:
- `browser_snapshot`
- `browser_tabs`
- `browser_console_messages`
- `browser_network_requests`
- `browser_take_screenshot`
- Class: `A0`

Navigation and bounded interaction:
- `browser_navigate`
- `browser_navigate_back`
- `browser_wait_for`
- `browser_hover`
- Class: `A1`

Potential state change:
- `browser_click`
- `browser_type`
- `browser_fill_form`
- `browser_select_option`
- `browser_press_key`
- `browser_drag`
- `browser_resize`
- Class: `A1` by default
- Escalate to `A3` if page context suggests submit/send/purchase/delete/account mutation

Explicit side-effect tools:
- `browser_file_upload`
- `browser_pdf_save`
- downloads into workspace
- Class: `A2` or `A3` depending on destination and remote impact

Dangerous browser infra:
- `browser_evaluate`
- `browser_run_code`
- Node DOM `delete` and `add` helper actions
- Class: `A4`
- Default: `blocked` outside explicit operator mode

## Sensitive Paths
These paths should escalate any write by at least one level:
- `.env`
- `.agent-state/`
- `memory/`
- `.github/workflows/`
- `.github/CODEOWNERS`
- `SECURITY.md`
- `RELEASE_CHECKLIST.md`
- credential files
- browser profile directories

## Policy Engine Placement
Target module layout:

```text
agent/agent/
  architecture/
    safety_types.py
    safety_policy.py
    safety_registry.py
    safety_audit.py
    safety_confirm.py
```

Responsibilities:
- `safety_types.py`: enums, decision/result structures
- `safety_registry.py`: metadata for each tool and action family
- `safety_policy.py`: decision logic and escalation rules
- `safety_confirm.py`: token issue/validate logic
- `safety_audit.py`: structured audit logging

## Integration Plan
### Phase 1
- Add tool metadata registry with action classes and policy tags.
- Wrap Python tool dispatch so every callable passes through policy evaluation before execution.
- Return `confirm_required` and `preview_required` JSON statuses instead of only `ok` or `blocked`.

### Phase 2
- Add preview support to filesystem writes and command execution.
- Add same-run confirmation token handling.
- Emit structured audit events.
- Add Node local browser-runtime gating for DOM mutation, uploads/downloads, and dangerous click intents.

### Phase 3
- Extend browser context escalation and confirmation to the raw MCP / VS Code surface, not only Python wrappers and local CLI runtime.
- Keep DOM mutation helpers blocked by default unless explicit operator mode is enabled.
- Add browser-side audit receipts and verifier summaries alongside the existing Python safety events.

### Phase 4
- Expose policy decisions in VS Code mode and Python mode UI.
- Add verifier receipts and trust summaries to final responses.

## Required Tests
### Unit tests
- action classification by tool and args
- confirmation token issue and expiry
- path sensitivity escalation
- overwrite escalation
- dangerous command blocking
- browser dangerous-action escalation

### Integration tests
- preview then confirm then execute for file write
- confirm token mismatch blocks execution
- destructive actions blocked without operator mode
- audit log emitted for `A1+` actions
- browser submit-like flow requires confirmation

## Immediate Repo Changes To Make Next
1. Introduce shared safety metadata for every tool in Python registry.
2. Add preview and confirm support to `fs_write`, `fs_copy`, `fs_move`, `fs_delete`, `fs_patch`, and `run_command`.
3. Block Node DOM `delete` and `add` helpers by default unless explicit operator mode is enabled.
4. Add a safety audit log under `.agent-state/`.
5. Update prompts and docs so the agent explains policy decisions instead of silently blocking.

## Current Status
- Python tool dispatch now enforces preview/confirm/blocked decisions with audit logging.
- Node local browser runtime now enforces same-run confirmation for risky actions through `browser-safety.js`, `edge-session.js`, and CLI `confirm` / `cancel` commands.
- Raw Playwright MCP traffic is now gated by a launcher-side JSON-RPC safety proxy that augments `tools/list` schemas and intercepts risky `tools/call` requests before they hit the child Playwright server.
- Raw MCP browser calls now append best-effort local safety audit events and proxy-generated evidence summaries to executed tool results.
- Raw MCP proxy verification now performs fresh post-action checks for navigation, tab actions, and screenshot/pdf artifact writes by issuing follow-up child MCP calls and filesystem existence checks.
- Raw MCP proxy execution now supports bounded retry/recovery for retryable browser actions and optional snapshot-hash-based verification for state-changing interactions.
- Raw MCP verification is still weaker than the Python path outside the proxy path: direct child MCP behavior without the launcher proxy still has no safety policy, retry budget, or structured audit/evidence layer.

## v1 Policy Summary
- Reads are cheap and automatic.
- Scoped reversible edits are allowed with verification.
- Broad or sensitive writes require preview first.
- External side effects require same-run confirmation.
- Destructive actions are blocked by default.
