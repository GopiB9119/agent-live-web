---
description: "Use when editing VS Code Copilot agent files, Playwright Edge MCP prompts, skills, instructions, or workflows in .github. Defines what the file is for, what it is not for, why, how, who, risks, bad impact, evidence, and validation."
applyTo: ".github/agents/*.agent.md,.github/prompts/*.prompt.md,.github/skills/web-works/*.md,.github/README.md,.github/copilot-instructions.md,.github/instructions/playwright-edge.instructions.md,.github/workflows/*.yml"
---

# Live Web Governance Structure

- Keep scope limited to VS Code Copilot agent behavior, Playwright Edge MCP, Playwright MCP, Playwright browser workflows, and live web autonomous execution.
- Do not mix Python runtime guidance into these files unless the file explicitly bridges both systems.
- Every prompt, skill, agent, instruction, or governance file in this lane should answer:
  - what it is for
  - what it is not for
  - why it exists
  - how it is used
  - who uses or owns it when relevant
  - bad impact if it is used incorrectly
  - evidence or validation expectations
- Prefer direct English, short bullets, and explicit trigger phrases.
- Keep one concern per file. If a rule is repo-wide, place it in `copilot-instructions.md`. If it is path-scoped, place it in `.github/instructions/`.
- Prompt files must stay single-purpose and parameterized.
- Skill files must define trigger, task intake, execution contract, verification, failure handling, and handoff.
- Agent files must define scope, non-goals, workflow, safety boundaries, and completion rules.
- Workflow files should have clear names and a readable purpose; add comments only when the logic is not obvious.
- When validation is possible, include exact commands or checks.