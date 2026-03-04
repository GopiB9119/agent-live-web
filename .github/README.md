# .github AI Governance Guide

This folder defines how Copilot and agents behave in this repository.

## Official behavior model
This layout follows current GitHub and VS Code guidance:
- Repo-wide instructions: `.github/copilot-instructions.md`
- Path-specific instructions: `.github/instructions/*.instructions.md` with `applyTo` frontmatter
- Custom agents: `.github/agents/*.agent.md`
- Agent memory/rules support: `AGENTS.md` at repo root

Important:
- VS Code can combine multiple instruction files for chat context.
- Instruction order is not guaranteed when multiple files apply.
- Keep files single-purpose and non-overlapping to avoid conflicts.

## Folder map in this repo
- `copilot-instructions.md`: global behavior, tool routing, execution contract
- `PROMPT_SECURITY_TEMPLATES.md`: reusable security/privacy/prompt-injection blocks
- `agents/agent-live-web.agent.md`: specialized live-web custom agent
- `instructions/*.instructions.md`: path-scoped implementation rules
- `instructions/README.md`: authoring and maintenance guide for scoped rules
- `skills/web-works/SKILL.md`: website workflow policy
- `skills/web-works/web-task.schema.json`: strict schema for JSON-driven web tasks
- `skills/web-works/web-task.template.json`: reusable template for website tasks
- `skills/web-works/PROMPTS.md`: prompt pack (master/resume/deep/turbo)

## Ownership model (single source per concern)
- Global behavior: `copilot-instructions.md`
- Security snippets: `PROMPT_SECURITY_TEMPLATES.md`
- Runtime constraints: `instructions/*.instructions.md`
- Task workflow: `skills/web-works/SKILL.md`
- JSON task contract: `skills/web-works/web-task.schema.json`

## Protected governance policy
- `.github/**` and `AGENTS.md` are protected by default during normal tasks.
- Agents should not modify governance files unless the user explicitly requests governance updates.

## When to edit which file
- Need global behavior change: edit `copilot-instructions.md`
- Need path-specific rule: add or edit `instructions/*.instructions.md`
- Need security prompt hardening: edit `PROMPT_SECURITY_TEMPLATES.md`
- Need different agent persona/tool list: edit `agents/*.agent.md`
- Need website workflow change: edit `skills/web-works/SKILL.md`
- Need JSON schema change: edit `skills/web-works/web-task.schema.json`

## Recommended workflow for web tasks
1. Copy `skills/web-works/web-task.template.json`.
2. Fill `start_url`, `objective`, `success_criteria`, and `steps`.
3. Use `site_profile: "generic"` for most websites and `site_profile: "whatsapp-web"` for WhatsApp Web.
4. Keep `auto_send_allowed: false` unless explicitly approved.
5. Execute one action per step with strict verification.

## Operator checklist
1. Use VS Code ownership mode (`vscode`) for browser work.
2. Validate task JSON against `web-task.schema.json`.
3. Run one tool action at a time and verify result.
4. Ask confirmation before irreversible side effects.
5. Report blockers with exact tool/selector/evidence.

## Official references
- GitHub custom instructions:
  - https://docs.github.com/en/copilot/how-tos/configure-custom-instructions/add-repository-instructions
- VS Code instructions model:
  - https://code.visualstudio.com/docs/copilot/customization/custom-instructions
- VS Code custom agents:
  - https://code.visualstudio.com/docs/copilot/customization/custom-agents
- OpenAI safety guidance:
  - https://developers.openai.com/apps-sdk/guides/security-privacy/
  - https://developers.openai.com/api/docs/guides/agent-builder-safety/
