---
name: web-works
description: Fast website understanding and execution skill for LLM agents with strict verification and optional JSON task contracts.
---

# web-works

## Purpose
Use this skill when user gives website work and expects fast, accurate execution.

This skill standardizes:
- site understanding
- selector strategy
- step-by-step verification
- JSON-driven automation inputs

## Trigger
Activate for requests like:
- "Open this site and do X"
- "Scrape/extract details from this web page"
- "Fill a form/download file/check UI state"
- "Use this JSON task and execute it"
- "Handle WhatsApp Web flow"

## Input modes
Two valid input modes are supported.

### Mode A: Natural language
Minimum required:
1. start URL
2. goal
3. success criteria

### Mode B: Structured JSON
Use:
- `.github/skills/web-works/web-task.schema.json`
- `.github/skills/web-works/web-task.template.json`
- `.github/skills/web-works/PROMPTS.md`

If JSON is provided, validate first and execute exactly by `steps`.

## Execution profiles
Use `execution_profile` from task JSON:
- `balanced`: default mode (`understand -> execute -> verify`)
- `deep`: force deeper analysis before risky/ambiguous steps
- `turbo`: after understanding is clear, reduce tool calls for speed

If task becomes ambiguous or fails repeatedly, switch to `deep` behavior temporarily even when profile is `balanced` or `turbo`.

## Cross-conversation resume protocol (required)
Use file-backed checkpoints so a new chat can continue from where previous chat stopped.

1. Resolve checkpoint path:
- Use `resume.state_file` from task JSON.
- If missing, default to `.playwright-mcp/resume-state.json`.
2. On new conversation with "continue/resume":
- Read checkpoint first.
- If checkpoint exists and `task_id` matches, resume from `last_completed_step_id + 1`.
- If checkpoint missing/mismatch, report it and restart from `start_url`.
3. Resume preflight (before next action):
- Verify active tab is real (not `about:blank`).
- Verify current URL/title still matches expected flow.
- Capture snapshot and re-resolve selectors (never reuse old element handles).
4. After each successful step, update checkpoint with:
- `task_id`
- `last_completed_step_id`
- `current_url`
- `updated_at`
- `status` (`running`/`blocked`/`done`)
5. On completion, write `status=done` and final evidence summary.

## Execution contract
For each step:
1. Plan one atomic action.
2. Execute one tool call.
3. Verify expected state.
4. Retry once with better selector/path.
5. Stop with blocker if second attempt fails.

Never claim done without evidence.

Honor these JSON toggles when provided:
- `reasoning.understand_first`
- `reasoning.auto_escalate_deep_research`
- `reasoning.max_retries_per_step`
- `response_quality.fix_grammar`
- `response_quality.strict_format`
- `response_quality.concise_step_reports`

## Fast website understanding pass
Run this before heavy actions:
1. Capture URL, title, page type.
2. Find primary nav and key CTA/actions.
3. Identify login wall vs main app shell.
4. Build shortest path to user goal.
5. Define verification signal for the next action before executing it.

## Local file understanding pass
If task includes files/commands, run this before edits/execution:
1. Confirm target path(s) exist.
2. Read relevant files before changing them.
3. Identify dependencies/config impacted by the change.
4. Define verification checks (tests/lint/output expectation).

Do not run write commands before this pass.

## Selector policy
Use selector fallback order:
1. role/aria
2. stable attributes (`data-*`, `id`, `name`)
3. label/placeholder
4. xpath
5. text-only fallback

Avoid generic `getByText("Search")` as first selector on dynamic apps.

## Evidence rules
Use concrete checks:
- navigation: URL and target element
- typing: field value updated
- click: resulting UI state change
- extraction: exact text and element context
- download: file path + extension + size > 0

## Side-effect safety
Require explicit confirmation before:
- send
- submit
- delete
- purchase
- merge
- push

For messaging apps, draft first and ask before send.

## WhatsApp Web profile
When `site_profile` is `whatsapp-web`:
1. Open `https://web.whatsapp.com`.
2. Wait for either app shell or QR login wall.
3. If QR wall present, pause for user login.
4. Search contact with robust selectors.
5. Open matching chat and verify header text.
6. Type message draft and verify compose box value.
7. Ask confirmation before send.

Use `web-task.template.json` and set:
- `site_profile` to `whatsapp-web`
- `start_url` to `https://web.whatsapp.com`
- `side_effect_policy.auto_send_allowed` to `false` unless user explicitly requests auto-send

## Output format per step
Use:
- `Action:`
- `Tool:`
- `Verification:`
- `Next:`

Also include:
- `Understanding:` one-line state summary before first action.

## Definition of done
Done only when:
- requested result is complete
- evidence is shown for critical steps
- no unresolved blocker remains
