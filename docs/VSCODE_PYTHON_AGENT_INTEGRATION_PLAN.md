# VS Code / Python Agent Integration Plan

## Goal
Add selected VS Code, Pylance, Playwright, and later GitHub-style capabilities to the Python agent without turning it into a flat list of hundreds of raw tools.

The Python agent should remain:
- planner
- router
- verifier
- safety gate
- audit/logging layer

The VS Code/Copilot-facing tool manifest in `.github/agents/agent-live-web.agent.md` should be treated as an upstream capability catalog, not as the Python agent's direct source of truth.

## Current Repo Fit
This repo already has the right long-term split:
- `agent/agent/agent.py` is the main control plane
- `agent/agent/tools.py` is the safety-wrapped execution layer
- `agent/agent/mcp_tools.py` is the browser/MCP bridge
- `.github/agents/agent-live-web.agent.md` is a broad VS Code-side capability catalog

The gap is not missing raw tools. The gap is controlled adoption:
- decide which capability surface matters for a task
- expose only those tools
- keep the shared preview/confirm/block policy
- verify outcomes before trusting the result

## Product Rule
The Python agent is the product brain.

VS Code/Copilot-style surfaces are capability providers.

That means:
- one shared planner and verifier
- one shared safety model
- multiple tool surfaces underneath
- no separate "Copilot brain" and "Python brain"

## Hard Rules
Do not expose the full raw tool list to the Python model.

Do not treat the VS Code tool manifest as the Python registry.

Do not let browser or GitHub tools become default for ordinary repo tasks.

Instead:
1. group tools by capability surface
2. expose only the needed surface for the current task
3. apply one shared safety model before execution
4. verify after each state-changing step
5. keep dangerous actions confirm-required or blocked

## Upstream Capability Catalog
The current `.github/agents/agent-live-web.agent.md` manifest is useful, but only as a catalog. It should be split into adoption buckets.

### Adopt early
These fit the Python-first product now:
- `read/readFile`
- `read/problems`
- `search/codebase`
- `search/textSearch`
- `search/usages`
- `search/listDirectory`
- `edit/editFiles`
- `edit/createFile`
- `edit/rename`
- `execute/runTests`
- `execute/getTerminalOutput`
- `execute/runInTerminal`
- `playwright-edge/browser_*` already aligned with MCP/browser runtime
- `pylance-mcp-server/pylanceSyntaxErrors`
- `pylance-mcp-server/pylanceFileSyntaxErrors`
- `pylance-mcp-server/pylanceImports`
- `pylance-mcp-server/pylancePythonEnvironments`
- `ms-python.python/getPythonEnvironmentInfo`
- `ms-python.python/getPythonExecutableCommand`

### Adopt later
These are valuable, but not part of the first serious MVP:
- `github/*` read flows
- PR comment/review flows
- `vscode/getProjectSetupInfo`
- `vscode/vscodeAPI`
- `execute/createAndRunTask`
- `pylance-mcp-server/pylanceInvokeRefactoring`
- `playwright-edge/browser_file_upload`
- `playwright-edge/browser_pdf_save`

### Keep behind stronger gates
These must never be default-exposed:
- `github/push_files`
- `github/merge_pull_request`
- `github/delete_file`
- `vscode/installExtension`
- `ms-python.python/installPythonPackage`
- `ms-python.python/configurePythonEnvironment`
- `playwright-edge/browser_run_code`
- destructive browser mutations

### Postpone
These are scope traps right now:
- `agent/runSubagent`
- `vscode/newWorkspace`
- notebook execution
- Bicep specialist tooling
- deep Copilot job management

## Recommended Capability Surfaces

### 1. Workspace surface
Map these first:
- `read/readFile`
- `read/problems`
- `search/codebase`
- `search/textSearch`
- `search/usages`
- `search/listDirectory`
- `edit/editFiles`
- `edit/createFile`
- `edit/rename`
- `execute/runTests`
- `execute/getTerminalOutput`
- `execute/runInTerminal`

Why first:
- highest value for normal developer tasks
- lowest product confusion
- strongest overlap with current Python agent mission

### 2. Playwright/browser surface
Map these next:
- `playwright-edge/browser_navigate`
- `playwright-edge/browser_tabs`
- `playwright-edge/browser_snapshot`
- `playwright-edge/browser_click`
- `playwright-edge/browser_type`
- `playwright-edge/browser_fill_form`
- `playwright-edge/browser_select_option`
- `playwright-edge/browser_wait_for`
- `playwright-edge/browser_take_screenshot`
- `playwright-edge/browser_pdf_save`
- `playwright-edge/browser_network_requests`
- `playwright-edge/browser_console_messages`

Why next:
- already aligned with current MCP/browser runtime
- critical for hybrid repo plus browser workflows

### 3. Python/Pylance surface
Map after workspace and browser:
- `pylance-mcp-server/pylanceSyntaxErrors`
- `pylance-mcp-server/pylanceFileSyntaxErrors`
- `pylance-mcp-server/pylanceImports`
- `pylance-mcp-server/pylancePythonEnvironments`
- `ms-python.python/getPythonEnvironmentInfo`
- `ms-python.python/getPythonExecutableCommand`
- `ms-python.python/configurePythonEnvironment`

Why:
- strong value for Python users
- useful for verification and environment debugging

### 4. GitHub/platform surface
Map later:
- issue/PR read
- search
- branch creation
- PR comment/review
- safe file updates

Do not start with:
- merge
- push
- delete repository/file
- assign Copilot jobs

### 5. Optional/narrow surfaces
Later only:
- notebook execution
- Bicep specialist tools
- extension installation
- subagents
- workspace creation

## Architecture Shape

### Keep
- `agent/agent/agent.py` as conversation/runtime entrypoint
- `agent/agent/tools.py` as high-level tool dispatch layer
- `agent/agent/tooling/registry.py` as schema/callable registry
- `agent/agent/mcp_tools.py` as current Playwright-specific bridge

### Add
Recommended new modules:

```text
agent/agent/
  capability_router.py
  task_spec.py
  surfaces/
    vscode_surface.py
    playwright_surface.py
    pylance_surface.py
    github_surface.py
    surface_registry.py
    surface_policy.py
```

### Responsibilities
- `task_spec.py`
  - normalize user task JSON / guided-form input
  - classify mode, risk, expected outputs
  - track no-submit and artifact requirements
- `capability_router.py`
  - choose which surface(s) to expose per task
  - keep model tool set small
  - prevent accidental GitHub/browser over-exposure
- `surfaces/surface_registry.py`
  - stable internal capability names
  - map external tool names to internal wrappers
  - keep upstream VS Code names separate from Python-facing aliases
- `surfaces/surface_policy.py`
  - read/write/destructive classes
  - submit/no-submit rules
  - confirmation gates
  - environment-sensitive policy such as local vs staging vs production-like
- `surfaces/vscode_surface.py`
  - editor, problems, file selection, terminal, tests
- `surfaces/playwright_surface.py`
  - wrap `mcp_tools.py`, not replace it all at once
- `surfaces/pylance_surface.py`
  - syntax/import/env/python diagnostics
- `surfaces/github_surface.py`
  - GitHub read/write workflows with strong confirmation rules

## Runtime Flow
For each task:

1. Parse user request or task JSON.
2. Build a normalized task spec.
3. Classify:
   - repo task
   - browser task
   - hybrid task
   - diagnostics task
   - platform task
4. Router selects surfaces.
5. Register only the needed tools for this run.
6. Planner creates small ordered steps.
7. Executor uses the smallest useful tool.
8. Verifier checks:
   - file changes
   - diagnostics/tests
   - browser state
   - expected outputs
   - artifacts requested in the task spec
9. If state-changing and risky:
   - preview
   - confirm
   - execute
   - verify
10. If verification fails:
   - stop
   - gather evidence
   - propose the next safe move

## Task Modes
The Python agent should classify tasks into explicit working modes before tool exposure.

Recommended modes:
- `inspect`
- `draft`
- `test`
- `mutate`
- `submit`
- `debug`

Suggested behavior:
- `inspect`: read/search/snapshot only
- `draft`: create or update files, but no external side effects
- `test`: run commands/tests/browser dry-runs, no submit
- `mutate`: workspace or browser state changes, but still no irreversible submit by default
- `submit`: only after explicit confirmation and stronger verification
- `debug`: collect traces, logs, diffs, diagnostics, screenshots, and retry within budget

## Task Spec
Support both:
- beginner guided input
- expert JSON input

Recommended JSON shape:

```json
{
  "task": "update a user in the admin portal and verify the result",
  "mode": "dry_run",
  "task_mode": "mutate",
  "developer_level": "standard",
  "workspace": {
    "target_files": ["tests/admin-user.spec.ts"],
    "focus_paths": ["src/", "tests/"]
  },
  "website": {
    "base_url": "https://admin.example.com",
    "environment": "staging",
    "auth_mode": "manual_login"
  },
  "operations": [
    { "action": "navigate", "target": "/users" },
    { "action": "search", "target": "user_table", "value": "john@example.com" },
    { "action": "update", "target": "status_field", "value": "active" }
  ],
  "validation": {
    "must_see": ["john@example.com", "active"],
    "must_not_submit_without_confirm": true,
    "expected_result": "User status visible as active"
  },
  "safety": {
    "allow_write": true,
    "allow_delete": false,
    "allow_submit": false,
    "data_sensitivity": "internal"
  },
  "artifacts": {
    "require_trace": true,
    "require_screenshot": true,
    "require_diff": true
  },
  "limits": {
    "retry_budget": 2,
    "stop_conditions": ["unexpected_login_wall", "prod_like_warning"]
  }
}
```

### Fields worth adding beyond the current draft
These are high-value and easy to miss:
- `task_mode`
- `environment`
- `data_sensitivity`
- `expected_result`
- `allowed_side_effects`
- `stop_conditions`
- `retry_budget`
- `rollback_hint`
- `artifacts_required`
- `no_submit`

These fields matter more than adding dozens of extra raw tools.

## Tool Exposure Strategy
Never expose all surfaces at once.

Recommended exposure rules:
- repo-only task: workspace plus diagnostics only
- browser-only task: Playwright plus minimal workspace
- hybrid task: workspace plus Playwright plus diagnostics
- Python task: workspace plus Pylance plus Python env tools
- GitHub task: GitHub tools only after explicit routing

This reduces:
- token waste
- wrong tool calls
- accidental risk
- planner confusion

## Verification Contract
Every serious task should end with evidence, not just a claim.

Required verifier outputs:
- `expected_result`
- `observed_result`
- `status`: passed, failed, partial, blocked
- `evidence`: tests, diagnostics, screenshot, trace, diff, terminal output
- `next_safe_step`

For browser tasks, prefer:
- current URL
- screenshot
- DOM snapshot or tab summary
- trace or network summary when relevant

For workspace tasks, prefer:
- changed files
- diff summary
- diagnostics summary
- test/lint/typecheck result

## Beginner vs Expert Intake
Do not force all users to write JSON.

### Beginner path
Use a guided form or a short sequence of questions:
- what do you want changed
- which website or app
- do you want inspect, draft, test, or real mutation
- is this local, staging, or production-like
- should the agent avoid submit/delete

### Expert path
Allow direct JSON task specs with strict schema validation.

The Python agent should normalize both into one internal `task_spec`.

## Risk Classes

### Safe by default
- read
- search
- diagnostics
- list
- inspect
- snapshot
- screenshot

### Preview or verify
- edit
- create
- rename
- run tests
- open new browser tab
- type into forms without submit

### Confirm required
- submit
- delete
- install
- package/environment change
- GitHub write
- browser uploads/downloads affecting local files
- browser code execution

### Block unless explicitly enabled
- repository destruction
- broad workspace rewrites
- production-site destructive mutation
- repo/branch deletion

## Copilot Coordination Stance
Treat Copilot as optional collaboration, not as the core runtime.

Good use:
- handoff prompts
- open-file/context preparation
- diff review support
- PR review assistance later

Bad use:
- depending on Copilot-private internals for core execution
- making trust depend on Copilot behavior
- treating Copilot as the primary safety model

## What Not To Do
- Do not import the raw Copilot/VS Code manifest as model-facing Python tools.
- Do not overload `mcp_tools.py` with unrelated VS Code/GitHub/Pylance logic.
- Do not allow GitHub write tools in the default tool set.
- Do not let browser tools become the default path for ordinary repo work.
- Do not make "all developer types" the first user target.

## Best First Users
Primary:
- developers maintaining internal/admin web apps
- developers writing or fixing Playwright flows
- developers debugging browser plus repo issues together

Secondary:
- Python developers using Pylance/env diagnostics
- QA/automation engineers

Poor first fit:
- arbitrary public web scraping
- uncontrolled consumer checkout flows
- multi-repo autonomous platform engineering

## First Serious MVP
The first serious version of this integration should be able to:
- understand repo plus editor context
- inspect diagnostics and tests
- generate or update Playwright or code files
- run dry-run verification without submit
- collect screenshot/trace/diff evidence
- ask before any meaningful side effect

It does not need:
- deep Copilot coordination
- all GitHub actions
- notebooks
- subagents
- arbitrary website autonomy

## Implementation Order

### Phase 1
- add `task_spec.py`
- add `capability_router.py`
- add `surfaces/vscode_surface.py`
- route repo/test/diagnostics tasks through a narrowed VS Code-style surface
- keep browser access through existing `mcp_tools.py`

### Phase 2
- add `surfaces/pylance_surface.py`
- add Python diagnostics and environment tools
- improve verifier receipts

### Phase 3
- split Playwright surface cleanly from generic MCP handling
- keep shared safety policy
- add dry-run-first browser workflows

### Phase 4
- add limited GitHub read and comment workflows
- keep write actions confirm-required

### Phase 5
- optional Copilot collaboration hooks
- only after the Python-first path is stable and verified

## Success Criteria
This phase is successful when:
- the Python agent can route tasks to the right surface
- the model never sees unnecessary tools
- risky actions remain behind the shared safety system
- browser plus workspace plus diagnostics workflows can be verified in one run
- the architecture stays understandable to contributors
- VS Code/Copilot-style capabilities strengthen the Python agent instead of fragmenting it
