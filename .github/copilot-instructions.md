# Copilot Repository Instructions

Apply these rules for all tasks in this repository.

## Scope and layering
- This file is the repository-wide baseline.
- Path-specific files in `.github/instructions/*.instructions.md` add rules when `applyTo` matches.
- Agent files in `.github/agents/*.agent.md` apply when that agent is selected.
- `AGENTS.md` may also apply based on runtime/client behavior.

Do not depend on instruction ordering. Keep rules non-overlapping and explicit.

## Conflict handling
- Safety/privacy/confirmation rules always win.
- For matched files, path-specific instructions can refine global defaults.
- If rules still conflict, choose the safer action and report the exact conflict.

## Core execution contract
- Stay inside user scope.
- Use one atomic tool call per step.
- Verify concrete state change after each step.
- Retry once with a better selector/path.
- If retry fails, stop and report blocker with exact tool/args/error.

## Understand-First Policy (required)
Before doing side effects, run a short understanding pass.

For website tasks:
1. Confirm URL and title.
2. Identify login wall vs usable app shell.
3. Identify primary regions and intended target element.
4. Define expected verification signal before action.

For local file/code tasks:
1. Confirm target paths exist.
2. Read relevant files before editing/running.
3. Identify dependencies/imports/config that can affect the change.
4. Define expected verification signal before action.

Do not execute side effects until this pass is complete.

## Tool routing
- Interactive website work: `playwright-edge/*`
- Static page retrieval: `web/fetch`
- Local files/code: read/search/edit/execute tools
- Repo operations (PR/issues/commits): `github/*` only when explicitly requested

Do not mix browser-control tools and `web/fetch` for the same step.

## Structured web task JSON
If user provides JSON text or a JSON file path:
1. Validate against `.github/skills/web-works/web-task.schema.json`.
2. If invalid, report exact missing/invalid fields and stop.
3. If valid, execute `steps` in order with per-step verification.
4. Respect `side_effect_policy` and `confirm_before`.

## Edge ownership
- Use one owner at a time:
  - `vscode` is the default and only supported owner in this repository.
  - VS Code MCP startup auto-claims `vscode` owner.
- If lock mismatch appears, stop and ask operator to switch owner.

## Selector policy
Use this order:
1. role/aria
2. stable attributes (`data-*`, `id`, `name`)
3. label/placeholder
4. xpath
5. text-only fallback

Avoid text-only selectors as first choice on dynamic sites.

## Adaptive Execution Modes
- `normal`: understanding + strict verification.
- `fast`: after understanding is confirmed, minimize tool calls and continue with stable selectors only.
- `deep-research`: if confused/ambiguous/failing after retry, collect more evidence (snapshot/source/structure), restate assumptions, then continue with safest path.

Never loop blindly. Switch to `deep-research` instead of repeating the same failed action.

## Side-effect and privacy policy
- Require explicit same-run confirmation before `send`, `submit`, `delete`, `purchase`, `merge`, `push`.
- For messaging workflows, draft first and confirm before send.
- Never expose secrets, tokens, cookies, credentials, or personal identifiers.
- Minimize data collection and redact sensitive data in logs/memory.

## Response Quality
- Always fix grammar and spelling in user-facing output.
- Keep technical values unchanged (`code`, paths, selectors, commands, URLs).
- Use clean, concise formatting with explicit action/result/verdict.
- If the user message is unclear, normalize it into a clear interpreted goal before execution.

## Real-time research policy
For "latest/current/today":
1. Use official primary sources first.
2. Cross-check with at least one independent source.
3. Return exact date context with confidence.
