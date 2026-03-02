# .github AI Governance Guide

This folder defines how Copilot and agents behave in this repository.

## Official behavior model
This layout follows current GitHub and VS Code guidance:
- Repo-wide instructions: `.github/copilot-instructions.md`
- Path-specific instructions: `.github/instructions/*.instructions.md` with `applyTo` frontmatter
- Custom agents: `.github/agents/*.agent.md`
- Agent memory/rules file support: `AGENTS.md` at repo level

Important:
- VS Code combines multiple instruction files for chat context.
- Order is not guaranteed when multiple instruction files apply.
- To avoid conflicts, keep each file single-purpose and non-overlapping.

## Folder map in this repo
- `copilot-instructions.md`: global behavior, tool routing, execution contract.
- `PROMPT_SECURITY_TEMPLATES.md`: reusable security/privacy/prompt-injection blocks.
- `agents/agent-live-web.agent.md`: specialized live-web custom agent.
- `instructions/*.instructions.md`: file/path scoped implementation rules.
- `instructions/README.md`: how to author and maintain path-scoped rules.
- `skills/web-works/SKILL.md`: reusable website workflow policy.
- `skills/web-works/web-task.schema.json`: strict schema for JSON-driven web tasks.
- `skills/web-works/web-task.template.json`: one template for all websites.
- `skills/web-works/PROMPTS.md`: reusable high-power prompt pack (master/resume/deep/turbo).

## Ownership model (single source per concern)
- Global behavior: `copilot-instructions.md`
- Security prompt snippets: `PROMPT_SECURITY_TEMPLATES.md`
- Runtime-specific constraints: `instructions/*.instructions.md`
- Task execution workflow: `skills/web-works/SKILL.md`
- Structured task payload contract: `web-task.schema.json`

## When to edit which file
- Need global rule for all requests: edit `copilot-instructions.md`
- Need rule only for certain paths/files: add or edit `instructions/*.instructions.md`
- Need stronger security template text: edit `PROMPT_SECURITY_TEMPLATES.md`
- Need different agent persona/toolset: edit `agents/*.agent.md`
- Need website task workflow changes: edit `skills/web-works/SKILL.md`
- Need JSON field/validation change: edit `web-task.schema.json`

## Recommended workflow for web tasks
1. Copy `skills/web-works/web-task.template.json`.
2. Fill `start_url`, `objective`, `success_criteria`, and `steps`.
3. Use `site_profile: "generic"` for normal sites, `site_profile: "whatsapp-web"` for WhatsApp.
4. Keep `auto_send_allowed: false` unless explicitly approved.
5. Execute with strict step verification.

## Operator checklist
1. Use VS Code ownership mode (`vscode`) for browser work.
2. Validate JSON task against `web-task.schema.json`.
3. Run one tool action per step and verify result.
4. Ask confirmation before irreversible side effects.
5. Report blockers with exact tool/selector/evidence.

## Official references
- GitHub custom instructions:
  - https://docs.github.com/en/copilot/how-tos/configure-custom-instructions/add-repository-instructions
- VS Code instructions model:
  - https://code.visualstudio.com/docs/copilot/customization/custom-instructions
- VS Code custom agents:
  - https://code.visualstudio.com/docs/copilot/customization/custom-agents
- OpenAI agent safety:
  - https://developers.openai.com/apps-sdk/guides/security-privacy/
  - https://developers.openai.com/api/docs/guides/agent-builder-safety/
