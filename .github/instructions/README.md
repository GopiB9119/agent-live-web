# Path-Specific Instructions

These files refine behavior for matching files using `applyTo` frontmatter.

## How this folder works
- Put narrow, implementation-specific rules here.
- Keep repository-wide rules in `.github/copilot-instructions.md`.
- Do not duplicate global rules unless a file-scope exception is required.
- Avoid overlapping `applyTo` patterns where possible.

## Current files
- `playwright-edge.instructions.md`
  - Scope: Edge MCP runtime JS/config files.
  - Purpose: owner lock, protocol cleanliness, runtime reliability.
- `python-agent.instructions.md`
  - Scope: `agent/agent/**/*.py` and `agent/agent/*.md`.
  - Purpose: MCP lifecycle, memory guarantees, local workspace behavior.
- `web-task-json.instructions.md`
  - Scope: website task JSON files.
  - Purpose: schema compliance, step design, side-effect safety.

## Authoring checklist
1. Add precise `applyTo` patterns.
2. Keep rules testable and short.
3. Include a validation command when practical.
4. If a rule belongs globally, move it to `.github/copilot-instructions.md` instead.
