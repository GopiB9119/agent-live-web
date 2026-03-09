# Roadmap and Status

This document explains what Agent Live Web is ready for now, what is still experimental, and what is intentionally postponed.

## Product Direction
Agent Live Web is a local-first developer agent for repo, terminal, editor, and browser workflows.

Primary product surface:
- Python agent

Supporting tool surfaces:
- VS Code MCP mode
- Playwright Edge MCP runtime
- local browser CLI/runtime

The design rule is simple:
- one main control plane
- multiple entry surfaces
- shared safety and verification model underneath

## Current State
The project is beyond prototype-level tooling, but it is not a fully polished general-purpose autonomous IDE agent yet.

Best description of current state:
- strong local developer-agent foundation
- real safety gating across Python, local browser, and MCP proxy paths
- solid unit baseline
- real Windows live MCP integration coverage
- packaging and onboarding improved, but still evolving

## Support Levels

### Stable Enough To Recommend
- Python agent as the main local workflow entrypoint
- `npm run verify` as the local baseline check
- shared preview / confirm / blocked safety model
- Python unit test suite
- JS safety/unit suite
- Windows live MCP integration workflow
- repo-local test/runtime directory strategy for Windows reliability

### Working But Still Sharp-Edged
- direct VS Code MCP mode
- direct MCP proxy path
- browser CLI/local runtime mode
- trace/session artifact debugging flows
- memory tooling
- raw MCP probe workflow

These work, but they still require a technically comfortable user when something goes wrong.

### Experimental
- broad autonomous multi-step workflows
- higher-level planner/executor/verifier evolution beyond current tool loop
- deeper recovery strategies
- advanced repo intelligence and ranking
- richer audit receipts and trust dashboard style summaries

### Not Ready / Intentionally Postponed
- deep Copilot coordination
- “whole IDE handles everything” autonomy
- broad long-term memory persistence by default
- multi-user/cloud-hosted orchestration
- aggressive self-directed background tasking

## What Is Strongest Right Now
- local-first orientation
- shared safety model
- browser/MCP verification depth
- Windows live MCP integration proof
- contributor baseline verification with one command

## What Is Weakest Right Now
- product polish for non-expert first-time users
- Python agent UX around model setup and MCP availability fallback
- direct VS Code surface still depends on user understanding MCP/runtime concepts
- roadmap and release communication still catching up to the architecture

## Near-Term Priorities

### Priority 1
- keep Python-first product path clear
- keep `npm run verify` and live MCP CI green
- make required GitHub checks match the verified baseline

### Priority 2
- improve troubleshooting and first-run safety messaging
- keep browser/runtime behavior aligned with the shared policy engine
- strengthen verification receipts and audit summaries

### Priority 3
- improve repo understanding and planning quality in the Python agent
- make hybrid repo-plus-browser workflows more deterministic
- reduce manual runtime knowledge needed for VS Code users
- adopt VS Code, Pylance, and Playwright capability surfaces through a routed Python-first architecture instead of a flat raw tool catalog

## Next Release Direction

### v5.1.x
- stabilize the current Python-first product shape
- keep safety gating and verification trustworthy
- tighten docs, CI, release discipline, and contributor experience

### v5.2
- continue MCP integration hardening
- improve verification and recovery behavior
- strengthen Python control-plane reliability around browser workflows

### Later
- stronger VS Code mode
- better planner/verifier architecture
- deeper repo intelligence
- limited, carefully bounded autonomy improvements
- optional Copilot collaboration after the Python-first routed-surface model is stable

## What Should Be Cut When Scope Expands
If scope starts drifting, cut these first:
- browser-first product messaging
- new autonomous features without verification
- more memory complexity without retention/privacy work
- Copilot coordination work before the main Python path is cleaner

## Trust Model
The goal is not “never fails.”

The goal is:
- fail rarely
- fail safely
- explain what happened
- preserve user control
- give enough evidence to debug the failure

Any roadmap item that weakens that tradeoff should be delayed.
