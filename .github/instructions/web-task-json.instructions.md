---
applyTo: ".github/skills/web-works/*.json,**/web-task*.json"
---

# Web Task JSON Instructions

## Goal
Keep website task JSON files strict, reusable, and easy for agents to execute.

## Rules
- Follow `.github/skills/web-works/web-task.schema.json`.
- Use explicit, measurable `success_criteria`.
- Keep `steps` atomic: one action per step.
- Add verification for every step.
- Set `execution_profile` explicitly (`balanced`, `deep`, or `turbo`) based on task complexity.
- Include `resume` block for cross-conversation continuation.
- Include `reasoning` and `response_quality` blocks for consistent behavior.
- Keep `auto_send_allowed` as `false` unless user explicitly requests auto-send.
- Use `confirm_before: true` for side-effect actions.

## Selector rules
- Prefer `css` with stable attributes (`data-*`, `id`, `name`).
- Use role/aria-style selectors where possible.
- Avoid brittle text-only selectors unless no alternative exists.

## WhatsApp-specific safety
- Default `site_profile` to `whatsapp-web` only for WhatsApp tasks.
- Always draft message first.
- Require confirmation before send.

## Validation checklist
1. Schema-valid JSON.
2. `start_url` and `allowed_domains` match.
3. Each step has `id`, `action`, and `verify`.
4. `resume.state_file` is present and writable in workspace.
5. `execution_profile`, `reasoning`, and `response_quality` are valid.
6. Side-effect policy is explicit.
7. Output format and fields are defined.
