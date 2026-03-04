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
If user gives website task JSON:
1. Validate against `.github/skills/web-works/web-task.schema.json`.
2. If invalid, report exact field errors and stop.
3. If valid, execute `steps` in order with verification.
4. Respect side-effect policy and confirmation flags.
