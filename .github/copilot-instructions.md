# Copilot Repository Instructions

Apply these rules across this repository.

## Core contract
- Stay inside user scope.
- Understand first, then execute.
- Run one atomic action per step.
- Verify each step before moving on.
- Retry once with a better selector/path; if it still fails, report blocker details.

## Tool routing
- Interactive websites: use Playwright Edge MCP tools.
- Static retrieval: use fetch-style tooling.
- Local code/files: use read/search/edit/execute tools.
- Do not mix multiple control paths for the same step.

## VS Code ownership
- Default browser owner is `vscode`.
- Keep one owner at a time for the shared browser profile.
- If owner lock mismatch appears, stop and surface the exact lock error.
- Detailed owner-lock and runtime rules: see `.github/instructions/playwright-edge.instructions.md`.

## Local-first privacy and security
- Prefer local runtime and local profile paths.
- Do not expose secrets, tokens, cookies, or credentials.
- Ask confirmation before irreversible actions (`send`, `submit`, `delete`, `purchase`, `merge`, `push`).
- Redact sensitive values in logs/reports.

## Governance file protection
- Treat `.github/**` and `AGENTS.md` as protected configuration.
- Do not edit protected files during normal execution tasks.
- Edit protected files only when the user explicitly requests governance/instruction changes.

## Structured website JSON behavior
- Website task JSON must validate against `.github/skills/web-works/web-task.schema.json`.
- Detailed task workflow: see `.github/skills/web-works/SKILL.md` and `.github/instructions/web-task-json.instructions.md`.

## Lane-specific rules
- **Live web / Playwright**: `.github/agents/agent-live-web.agent.md`, `.github/instructions/playwright-edge.instructions.md`, `.github/skills/web-works/SKILL.md`
- **Python agent**: `.github/instructions/python-agent.instructions.md`
- **Governance authoring**: `.github/instructions/live-web-governance.instructions.md`

Do not duplicate lane-specific rules here. Keep this file short and global.
