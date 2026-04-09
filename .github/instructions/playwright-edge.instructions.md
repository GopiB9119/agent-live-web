---
applyTo: "playwright-edge-mcp.js,edge-session.js,cli-agent.js,nl-command-parser.js,scripts/**/*.js,.vscode/mcp.json"
---

# Playwright Edge Runtime Instructions

## What This File Is For
- Playwright Edge MCP runtime files, CLI browser runtime files, session logic, parser logic, support scripts, and MCP config that drive live web execution.
- Runtime safety, owner lock stability, deterministic browser behavior, and evidence-backed recovery.

## What This File Is Not For
- Python runtime-agent rules.
- Repo-wide Copilot behavior that belongs in `.github/copilot-instructions.md`.
- Website task JSON schema rules or generic prompt wording.

## Why These Rules Exist
- A small mistake in owner locking, stdout handling, retry logic, selectors, or runtime defaults can break the whole live-web workflow.
- The VS Code Copilot live-web lane depends on deterministic runtime behavior and accurate verification signals.

## Who This File Protects
- The VS Code Copilot operator using the live-web lane.
- The shared browser profile owner `vscode`.
- Downstream prompts, skills, and workflows that assume the runtime is stable.

## Runtime Requirements
- Preserve owner lock behavior (`PLAYWRIGHT_MCP_OWNER`, owner file, lock file).
- Keep `vscode` as the default owner for this repository.
- Do not remove single-owner enforcement unless explicitly asked.
- Keep MCP stdout protocol clean; write diagnostics to stderr.
- Prefer configurable env vars for timeouts and runtime toggles.
- Keep defaults performance-oriented but stable.

## What Must Not Happen
- Avoid broad command-line relaxations that reduce security.
- Keep blocked origins support in place unless user explicitly asks to disable.
- Do not add automatic destructive process-kill logic.
- Do not silently change the default owner, owner-file behavior, or shared-profile rules.
- Do not weaken verification so a browser action can report success without evidence.

## Dynamic UI Reliability Constraints
- Before click/type actions in automation flows, enforce preconditions: no active blocker overlay, no unresolved onboarding gate, target is visible and enabled.
- Prefer visible role-based targeting scoped to active containers (`main`, active dialog, chat panel) over global selectors.
- For chatbot turns, require two success signals in logic and tests: user-send confirmed and a new assistant response confirmed.
- Extract chatbot output from the latest assistant message node only; avoid broad DOM text scraping for result evaluation.
- On blocked interactions, follow one retry ladder only: `Escape` → neutral outside click → explicit close control → re-verify → retry with tighter scope.

## Bad Impact To Avoid
- Wrong browser owner or corrupted shared profile access.
- Broken MCP protocol because diagnostics leak to stdout.
- Selector drift that clicks or types into the wrong target.
- False success states after a failed or ambiguous browser action.
- Runtime changes that silently weaken safety or side-effect controls.

## Evidence Requirements
- When changing owner or session logic, state the expected owner-lock and profile effect.
- When changing parser or session behavior, describe the exact success signal and recovery path.
- When changing selector or chatbot logic, ensure tests cover both success and failure or ambiguity.
- When changing scripts under `scripts/**/*.js`, explain whether the change affects operator workflow, runtime safety, or diagnostics.

## Validation After Edits
- `node --check playwright-edge-mcp.js`
- For parser/session changes:
  - `node --check edge-session.js`
  - `node --check nl-command-parser.js`
  - `node --check cli-agent.js`
- For behavior changes in session or CLI flow:
  - `npm run test:js:unit`
- For broader runtime changes:
  - `npm run check`
