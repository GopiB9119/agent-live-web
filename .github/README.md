# .github AI Governance Guide

This folder controls how VS Code Copilot, custom agents, prompts, skills, instructions, and workflows behave in this repository.

## Main lanes

### 1. Live Web Copilot lane
Use this lane for VS Code Copilot work that depends on Playwright Edge MCP, Playwright MCP, browser automation, or live web autonomous execution.

Primary files:
- `agents/agent-live-web.agent.md`
- `prompts/playwright-live-web-task-brief.prompt.md`
- `skills/web-works/SKILL.md`
- `skills/web-works/PROMPTS.md`
- `instructions/playwright-edge.instructions.md`
- `instructions/live-web-governance.instructions.md`

### 2. Repo-wide governance lane
Use this lane for rules that apply across the whole repository.

Primary files:
- `copilot-instructions.md`
- `PROMPT_SECURITY_TEMPLATES.md`

### 3. Runtime-specific lanes
Use these when the rule belongs to one implementation surface only.

Primary files:
- `instructions/playwright-edge.instructions.md`
- `instructions/python-agent.instructions.md`
- `instructions/web-task-json.instructions.md`

### 4. Workflow and CI lane
Use this lane for automation and validation around the customization surface.

Primary files:
- `workflows/*.yml`

## What belongs where
- Need repo-wide behavior change: edit `copilot-instructions.md`
- Need the custom agent behavior or persona: edit `agents/agent-live-web.agent.md`
- Need a structured intake prompt for live-web work: edit `prompts/*.prompt.md`
- Need website workflow policy: edit `skills/web-works/SKILL.md`
- Need JSON task contract: edit `skills/web-works/web-task.schema.json`
- Need path-scoped implementation rules: edit `instructions/*.instructions.md`
- Need CI validation or enforcement: edit `workflows/*.yml`

## Live Web task brief contract
For VS Code Copilot + Playwright work, define these fields before execution:
1. What you want.
2. What you do not want.
3. Why it matters.
4. Which surface is in scope.
5. Who is acting or providing auth context.
6. How to work (`explore`, `extract`, `automate`; `balanced`, `deep`, `turbo`).
7. Bad impact to avoid.
8. Evidence required.
9. Stop-and-ask actions.
10. Done condition.

The prompt file `prompts/playwright-live-web-task-brief.prompt.md` exists to enforce this intake shape.

## What not to do
- Do not duplicate the same rule across `copilot-instructions.md`, agent files, and skill files.
- Do not mix Python runtime guidance into the live-web lane unless the file is explicitly bridging both systems.
- Do not broaden `applyTo` patterns without a strong reason.
- Do not create multi-purpose prompt files when the task needs a skill or agent instead.
- Do not leave grammar, purpose, or validation paths vague in governance files.

## Protected governance policy
- `.github/**` and `AGENTS.md` are protected during normal execution work.
- Edit them only when the user explicitly requests governance or customization changes.

## Recommended flow for live-web governance changes
1. Start with the structured task brief.
2. Decide whether the change belongs in an agent, prompt, skill, instruction, or workflow file.
3. Keep one concern per file.
4. State purpose, non-purpose, why, how, who, bad impact, and validation.
5. Validate matching runtime files when the change affects executable behavior.

## References
- GitHub custom instructions:
  - https://docs.github.com/en/copilot/how-tos/configure-custom-instructions/add-repository-instructions
- VS Code custom instructions:
  - https://code.visualstudio.com/docs/copilot/customization/custom-instructions
- VS Code custom agents:
  - https://code.visualstudio.com/docs/copilot/customization/custom-agents
- VS Code prompt files:
  - https://code.visualstudio.com/docs/copilot/customization/prompt-files