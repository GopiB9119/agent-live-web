# Modes and Supported Workflows

Use this document to decide which entrypoint to use.

Default rule:
- start with `npm run agent:python`
- only switch to VS Code MCP mode or local browser CLI mode when the task actually needs that surface

## 1. Mode chooser

### Python agent mode
Command:
```bash
npm run agent:preflight
npm run agent:mcp-status
npm run agent:python
```

Use this for:
- repo understanding
- file/code changes
- tests, diagnostics, and command workflows
- hybrid tasks that need both repo work and browser verification
- most normal day-to-day use
- quick local readiness checks before starting the chat loop

Session behavior:
- run `npm run agent:preflight` first when you want readiness without starting a resumable session
- run `npm run agent:mcp-status` when you only want MCP proxy readiness/trust without opening chat
- session auto-resume stays locked until one normal assistant turn completes
- `/save` before that point is expected to refuse persistence
- `/reset` clears stale resume state when the current run is still untrusted
- `/mcp` shows the current MCP proxy readiness/trust report when browser tools are connected

Do not choose this first when:
- you only need direct VS Code MCP/browser control
- you are debugging the raw browser runtime itself

### VS Code MCP mode
Command:
```bash
npm run agent:vscode
```

Use this for:
- direct VS Code browser/MCP workflows
- shared `vscode` browser ownership
- editor-side or MCP-panel driven browser work
- validating direct MCP behavior
- inspecting direct proxy readiness with the `agent_proxy_status` MCP tool

Do not choose this first when:
- the task is mainly repo/code work
- you want the main product path
- you do not need direct browser/editor control

### Local browser CLI mode
Command:
```bash
npm run agent:live-web
```

Use this for:
- local browser runtime debugging
- lower-level browser command testing
- operator/debug work on browser actions

Do not choose this first when:
- you want a general developer agent workflow
- the task is mostly code, repo, or test work

## 2. Official supported workflows

### Workflow A: Repo-first implementation
Recommended mode:
- `npm run agent:python`

Examples:
- inspect codebase and explain architecture
- edit files and run tests
- diagnose failing unit tests
- refactor a module with verification

### Workflow B: Hybrid repo plus browser verification
Recommended mode:
- `npm run agent:python`

Examples:
- change code, then verify behavior through MCP/browser tools
- debug a browser-facing issue that also needs repo edits
- collect browser evidence for a code/runtime change

### Workflow C: Direct VS Code browser work
Recommended mode:
- `npm run agent:vscode`

Examples:
- use the MCP panel directly
- validate browser actions in the shared VS Code owner flow
- reproduce browser-side issues without the full Python agent loop
- inspect proxy safety/readiness via `agent_proxy_status`

### Workflow D: Raw MCP/runtime debugging
Recommended mode:
- `npm run agent:vscode`
- `npm run mcp:probe`

Examples:
- isolate MCP startup failures
- inspect owner-lock/runtime issues
- compare launcher path vs direct Playwright MCP path

### Workflow E: Local browser runtime/operator debugging
Recommended mode:
- `npm run agent:live-web`

Examples:
- debug local browser action handling
- inspect CLI/runtime command behavior
- test browser action confirmation and blocking behavior

## 3. Supported vs not-yet-supported expectations

### Supported now
- Python-first local developer workflows
- shared safety gating across Python, local browser, and MCP proxy paths
- repo plus browser hybrid verification flows
- Windows live MCP integration verification

### Working but sharp-edged
- direct VS Code MCP workflows
- raw MCP/runtime debugging
- local browser CLI/operator flows

### Not the current product promise
- whole-IDE autonomous orchestration for everything
- deep Copilot coordination
- broad background autonomy
- “never fails” agent behavior

## 4. Simple decision table
- Mostly code, files, tests, or repo reasoning: `npm run agent:python`
- Code change plus browser verification: `npm run agent:python`
- Direct VS Code MCP/browser flow: `npm run agent:vscode`
- Browser runtime or CLI debugging: `npm run agent:live-web`
- MCP startup investigation: `npm run mcp:probe`

## 5. If you chose the wrong mode
- If you started in VS Code MCP mode but the task is mostly repo/code work:
  - switch to `npm run agent:python`
- If you started in Python mode but the issue is direct MCP startup/runtime:
  - use `npm run agent:vscode` or `npm run mcp:probe`
- If you hit owner-lock or runtime problems:
  - read `docs/TROUBLESHOOTING.md`

## 6. Related docs
- `docs/ARCHITECTURE.md`
- `docs/ROADMAP.md`
- `docs/TROUBLESHOOTING.md`
- `docs/SAFETY_GATING_DESIGN.md`

## 7. Trusted Session Rules
- first-run readiness checks are allowed, but they do not unlock session auto-resume
- immediate quit does not unlock session auto-resume
- timeout-only or tool-churn-only runs do not unlock session auto-resume
- one completed normal assistant turn unlocks resumable session state for later runs
