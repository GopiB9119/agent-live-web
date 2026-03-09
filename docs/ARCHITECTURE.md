# Architecture

This document explains the current high-level architecture of Agent Live Web.

Use this when you need to understand:
- which entrypoint to use
- how the Python agent, MCP runtime, and browser layer fit together
- where major responsibilities live in the repo

For deeper design details:
- safety model: `docs/SAFETY_GATING_DESIGN.md`
- Python refactor plan: `agent/agent/ARCHITECTURE_PLAN.md`
- VS Code/Python capability expansion: `docs/VSCODE_PYTHON_AGENT_INTEGRATION_PLAN.md`
- status and maturity: `docs/ROADMAP.md`

## 1. Top-Level Shape
Agent Live Web is built as one main control plane plus supporting tool surfaces.

Primary control plane:
- Python agent

Supporting surfaces:
- VS Code MCP mode
- Playwright Edge MCP launcher/runtime
- local browser CLI/runtime

Design intent:
- one shared safety model
- one shared browser/MCP infrastructure
- one main agent path for repo and hybrid workflows

## 2. Official Entry Points

### Python agent
Use for:
- repo understanding
- code/file changes
- command/test workflows
- hybrid repo plus browser tasks

Command:
```bash
npm run agent:python
```

Main file:
- `agent/agent/agent.py`

### VS Code MCP mode
Use for:
- direct VS Code browser/MCP workflows
- shared browser ownership with the `vscode` owner

Command:
```bash
npm run agent:vscode
```

Main files:
- `playwright-edge-mcp.js`
- `.vscode/mcp.json`

### Local browser CLI/runtime
Use for:
- lower-level browser workflow testing
- local runtime debugging

Command:
```bash
npm run agent:live-web
```

Main files:
- `cli-agent.js`
- `edge-session.js`

## 3. Layer Model

### Layer A: Interfaces
These are the user-facing surfaces.

Files:
- `agent/agent/agent.py`
- `.vscode/mcp.json`
- `cli-agent.js`

Responsibilities:
- accept tasks
- start the right runtime path
- report results back to the user

### Layer B: Python agent control plane
This is the main product brain today.

Files:
- `agent/agent/agent.py`
- `agent/agent/tools.py`
- `agent/agent/tooling/registry.py`
- `agent/agent/tooling/schemas.py`

Responsibilities:
- manage the conversation loop
- expose tools to the model
- dispatch tool calls
- route all Python tool execution through the safety gate

### Layer C: Tool managers
These are the domain-specific capability modules behind the Python agent.

Files:
- `agent/agent/fs_tools.py`
- `agent/agent/command_tools.py`
- `agent/agent/memory_tools.py`
- `agent/agent/web_tools.py`
- `agent/agent/oauth_tools.py`
- `agent/agent/mcp_tools.py`
- `agent/agent/workflow_tools.py`
- `agent/agent/diagnostics_tools.py`
- `agent/agent/runtime_utils.py`

Responsibilities:
- filesystem reads/writes
- command execution
- memory and retrieval
- web fetch and OAuth
- MCP lifecycle and browser wrappers
- workflow planning/execution helpers
- diagnostics and health reporting

### Layer D: Shared safety layer
This is the cross-surface guardrail system.

Python files:
- `agent/agent/architecture/safety_types.py`
- `agent/agent/architecture/safety_registry.py`
- `agent/agent/architecture/safety_confirm.py`
- `agent/agent/architecture/safety_policy.py`
- `agent/agent/architecture/safety_audit.py`

Node/browser files:
- `browser-safety.js`
- `mcp-safety-adapter.js`

Responsibilities:
- classify actions
- decide allow / verify / preview / confirm / block
- mint and validate confirmation tokens
- write safety audit events
- keep Python, local browser, and MCP proxy behavior aligned

### Layer E: Browser/MCP infrastructure
This is the runtime layer for browser automation.

Files:
- `playwright-edge-mcp.js`
- `playwright-mcp-launch-config.js`
- `mcp-jsonrpc-transport.js`
- `mcp-safety-adapter.js`
- `edge-session.js`
- `browser-safety.js`
- `scripts/mcp-raw-probe.js`
- `scripts/mcp-init-page.js`

Responsibilities:
- launch Playwright MCP child server
- speak stdio MCP transport
- proxy and gate tool calls
- verify browser-side results
- manage owner lock and browser profile state
- provide local probe/debug paths

## 4. Current Runtime Flow

### Python agent flow
1. User starts `npm run agent:python`.
2. `agent.py` loads model/provider config and session state.
3. `agent.py` initializes MCP connectivity through `init_mcp_client()` when available.
4. The model receives tool schemas from the Python tool registry.
5. Tool calls route through `execute_tool_with_policy(...)` in `tools.py`.
6. The safety layer decides whether the action is allowed, needs verification, preview, confirmation, or is blocked.
7. If allowed, the domain manager executes the tool.
8. The result is returned with safety/audit/verification metadata where applicable.
9. The agent continues until it can produce a final answer.

### Direct MCP flow
1. User starts VS Code MCP mode or launches `playwright-edge-mcp.js`.
2. The launcher starts the child Playwright MCP server.
3. `mcp-jsonrpc-transport.js` handles stdio message parsing/encoding.
4. `mcp-safety-adapter.js` intercepts risky tool calls before the child executes them.
5. The proxy may return `preview_required`, `confirm_required`, or `blocked`.
6. Executed calls are augmented with safety, evidence, retry, and verification metadata.

### Local browser CLI flow
1. User starts `npm run agent:live-web`.
2. `cli-agent.js` parses local browser commands/intents.
3. `edge-session.js` drives browser actions.
4. `browser-safety.js` applies the same decision model for risky browser actions.

## 5. Module Map

### Core files
- `agent/agent/agent.py`: conversation loop and main Python entrypoint
- `agent/agent/tools.py`: Python tool dispatch and safety-wrapped execution
- `agent/agent/tooling/registry.py`: callable registration
- `agent/agent/tooling/schemas.py`: model-facing tool schemas

### Safety files
- `agent/agent/architecture/*`
- `browser-safety.js`
- `mcp-safety-adapter.js`

### Browser/MCP files
- `playwright-edge-mcp.js`
- `playwright-mcp-launch-config.js`
- `mcp-jsonrpc-transport.js`
- `edge-session.js`
- `.vscode/mcp.json`

### Test and verification files
- `agent/agent/tests/`
- `agent/agent/tests/integration/`
- `tests/`
- `.github/workflows/python-agent-tests.yml`
- `.github/workflows/mcp-live-integration.yml`

## 6. Current Responsibility Boundaries

### Stable boundaries
- Python agent is the main product surface.
- Safety policy is shared across Python, local browser, and MCP proxy paths.
- Browser runtime is infrastructure, not the primary product.

### Boundaries still evolving
- planner/executor/verifier behavior inside the Python agent is still lighter than the long-term design
- VS Code mode is useful, but still closer to a controlled tool surface than a full polished standalone agent mode
- direct MCP usage is stronger than before, but still less friendly than the main Python path

## 7. What To Read Next
- Want safety rules: `docs/SAFETY_GATING_DESIGN.md`
- Want maturity/status: `docs/ROADMAP.md`
- Want troubleshooting: `docs/TROUBLESHOOTING.md`
- Want Python refactor details: `agent/agent/ARCHITECTURE_PLAN.md`
- Want VS Code/Pylance/Playwright adoption plan for the Python agent: `docs/VSCODE_PYTHON_AGENT_INTEGRATION_PLAN.md`
