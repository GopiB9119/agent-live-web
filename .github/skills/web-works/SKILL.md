---
name: web-works
description: Use when Playwright Edge MCP, Playwright MCP, live web autonomous workflows, or structured website tasks need fast understanding, strict verification, and controlled side effects.
---

# web-works

## Purpose
Use this skill for the VS Code Copilot live-web lane when the task involves websites, browser automation, or a structured Playwright workflow.

This skill standardizes:
- task intake
- site understanding
- selector strategy
- one-step execution
- strict verification
- safe side-effect handling
- structured handoff

## Use this skill for
- opening and navigating websites or dashboards
- extracting details from live pages
- filling forms, downloading files, or checking UI state
- Playwright Edge MCP or Playwright MCP execution
- JSON-driven website tasks
- hybrid flows where browser work and local file work support one live-web goal

## Do not use this skill for
- unrelated Python runtime-agent work
- generic repo refactors with no browser or live-web goal
- irreversible side effects without same-run confirmation

## Required task brief
Before execution, convert the request into:
1. What the user wants.
2. What the user does not want.
3. Why the task matters.
4. Which site, app, or workflow is in scope.
5. Who is acting or supplying auth context.
6. How to work: `explore`, `extract`, or `automate`; `balanced`, `deep`, or `turbo`.
7. Bad impact to avoid.
8. Evidence required.
9. Stop-and-ask actions.
10. Done condition.

If the request is missing fields, infer the safest reversible assumption or ask one short question.

## Input modes

### Mode A: Natural language
Minimum required:
1. start URL or target website
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
- `deep`: force deeper analysis before risky or ambiguous steps
- `turbo`: after understanding is clear, reduce tool calls for speed

If the task becomes ambiguous or fails repeatedly, switch to `deep` behavior temporarily even when profile is `balanced` or `turbo`.

## Cross-conversation resume protocol
Use file-backed checkpoints so a new chat can continue from where a previous chat stopped.

1. Resolve checkpoint path.
2. On resume, read checkpoint first.
3. Verify current tab, URL, and title before the next action.
4. Re-resolve selectors from the current DOM.
5. Update the checkpoint after each successful step.
6. Mark `done` only after final evidence is captured.

## Execution contract
For each step:
1. Plan one atomic action.
2. Execute one tool call.
3. Verify expected state.
4. Retry once with a better selector or path.
5. Stop with blocker details if the second attempt fails.

Never claim completion without evidence.

Honor these JSON toggles when provided:
- `reasoning.understand_first`
- `reasoning.auto_escalate_deep_research`
- `reasoning.max_retries_per_step`
- `response_quality.fix_grammar`
- `response_quality.strict_format`
- `response_quality.concise_step_reports`

## Fast website understanding pass
Run this before heavy actions:
1. Capture URL, title, and page type.
2. Find primary navigation and key CTA surfaces.
3. Identify login wall, onboarding wall, or main app shell.
4. Build the shortest safe path to the goal.
5. Define the verification signal for the next action before executing it.

## Local file understanding pass
If the task also includes files or commands:
1. Confirm target paths exist.
2. Read relevant files before editing.
3. Identify impacted config or runtime surfaces.
4. Define validation before the first write.

## Selector policy
Use selector fallback order:
1. role or aria
2. stable attributes (`data-*`, `id`, `name`)
3. label or placeholder
4. xpath
5. text-only fallback

Avoid generic text-only selectors as the first choice on dynamic apps.

## Evidence rules
Use concrete checks:
- navigation: URL and target element
- typing: field value updated
- click: resulting UI state change
- extraction: exact text and element context
- download: file path, extension, and size > 0

## Failure and bad-impact control
Primary bad impacts:
- wrong site or page
- wrong account or auth state
- wrong element interaction
- unintended send, submit, or purchase
- wrong summary with weak evidence

After the first failure:
1. Freeze repeated actions.
2. Capture URL, title, visible state, and likely blocker.
3. Change one variable.
4. Retry once.
5. Stop with blocker if it still fails.

## Side-effect safety
Require explicit confirmation before:
- send
- submit
- delete
- purchase
- merge
- push

For messaging apps, draft first and ask before send.

## Output format per step
Use:
- `Understanding:`
- `Action:`
- `Tool:`
- `Verification:`
- `Next:`

## Definition of done
Done only when:
- the requested result is complete
- evidence is shown for critical steps
- no unresolved blocker remains
- the handoff is clear and actionable