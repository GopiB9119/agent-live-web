# Trust and Reliability Execution Plan

## Objective
Solve browser-action trust problems one by one until the system is safe, predictable, and explainable enough for daily developer use.

This plan is intentionally sequential.
Do not try to fix every problem in one large refactor.
Each phase should end with:
- clear code changes
- tests or verification checks
- updated docs
- a yes or no decision on whether the phase is complete

## Problems We Must Solve
These are the main problem types found in the current repo:

1. Wrong actions in a real browser
- the system can click, type, delete, or inject into the wrong target

2. Fragile automation on changing UI
- selectors can become ambiguous or unstable
- DOM fallback can hit the wrong element when layout changes

3. Unclear product behavior
- the JS paths and Python paths do not follow the same action contract
- users cannot always tell what the system decided to do and why

4. Lack of trust when the tool does not verify what it did
- successful execution is sometimes treated as success even when the intended outcome was not proven

5. Secret and sensitive data exposure
- logs, traces, failure artifacts, tool transcripts, and saved evidence must not leak keys, cookies, auth headers, or secret-bearing URLs

6. Weak evidence after failures
- when something goes wrong, developers need enough evidence to debug and trust the next run

## Planning Rules
Use these rules for every phase:

1. One problem class per phase
- do not mix product cleanup, selector redesign, and runtime migration in the same implementation pass

2. Shared contract first, then convenience features
- correctness and trust are more important than adding new commands

3. Keep one safest path
- once a safer runtime path exists, legacy paths should either route through it or be clearly deprecated

4. Every state-changing action must be explainable
- what target was selected
- what action was attempted
- what verification was used
- what evidence was captured

5. No silent success
- if the tool cannot verify the outcome, it should report uncertainty instead of success

## Work Types and Phase Order

### Phase 0: Baseline and Safety Gate
Goal:
Create the baseline checks needed before changing core browser behavior.

Why first:
Without a baseline, later changes will feel correct but can regress silently.

Primary files:
- [agent/agent/tests/test_mcp_tools.py](agent/agent/tests/test_mcp_tools.py)
- [agent/agent/tests/integration/test_mcp_live_integration.py](agent/agent/tests/integration/test_mcp_live_integration.py)
- [scripts/health-snapshot.js](scripts/health-snapshot.js)
- [README.md](README.md)

Tasks:
- keep the current MCP unit and live integration coverage healthy
- define a standard verification checklist for browser behavior changes
- make sure repo contributors use the repo-local `.venv` for Python verification

Done when:
- the current browser safety tests are stable
- contributors have one documented verification path for JS and Python browser changes

Verification:
- `.\\.venv\\Scripts\\python.exe -m unittest discover -s agent/agent/tests -p "test_*.py" -v`
- `.\\.venv\\Scripts\\python.exe -m unittest discover -s agent/agent/tests/integration -p "test_*.py" -v`

### Phase 1: Remove or Contain Unsafe Legacy Entry Points
Goal:
Stop unsafe direct browser control paths from bypassing the safer runtime behavior.

Problem being solved:
- wrong actions from raw Playwright calls without verification or ownership control
- resource leaks from unmanaged browser lifecycle

Primary files:
- ~~agent-logic.js~~ (removed in v6.0)
- ~~playwright-edge-agent.js~~ (removed in v6.0)
- [cli-agent.js](cli-agent.js)
- [README.md](README.md)

Tasks:
- ~~classify `agent-logic.js` and `playwright-edge-agent.js` as legacy, deprecated, or internal-only~~ **DONE: removed in v6.0**
- ~~decide whether these files should be removed, wrapped, or rerouted through [edge-session.js](edge-session.js)~~ **DONE: removed**
- ensure browser lifecycle cleanup is not optional in active code paths
- update docs so developers are not guided into the unsafe path by accident

Done when:
- there is one clearly recommended JS browser runtime path
- legacy helpers no longer bypass the safer action model in normal usage

Verification:
- `npm run check`
- manual smoke test through the supported CLI or MCP path only

### Phase 2: Unify the Action Contract Across Runtimes
Goal:
Make JS and Python browser actions return the same style of result and verification.

Problem being solved:
- unclear product behavior
- inconsistent trust level between runtimes

Primary files:
- [edge-session.js](edge-session.js)
- [agent/agent/mcp_tools.py](agent/agent/mcp_tools.py)
- [cli-agent.js](cli-agent.js)
- [docs/PRODUCT_ARCHITECTURE.md](docs/PRODUCT_ARCHITECTURE.md)

Tasks:
- define one browser action result contract for both runtimes
- include action name, target, attempts, verification result, evidence, error, and recovery status
- align JS output with the stronger Python MCP output model
- make CLI output easier to inspect and trust

Done when:
- the same type of browser action produces comparable evidence in JS and Python paths
- contributors can reason about one action format instead of multiple styles

Verification:
- add or update tests for structured action responses
- run CLI smoke actions and confirm the output includes verification details

### Phase 3: Strengthen Post-Action Verification
Goal:
Verify outcomes, not just successful API calls to Playwright.

Problem being solved:
- lack of trust when the tool says success without proving the intended effect

Primary files:
- [edge-session.js](edge-session.js)
- [agent/agent/mcp_tools.py](agent/agent/mcp_tools.py)

Tasks:
- after fill and type, read back the field value or content
- after click, verify URL change, state change, or target state change where possible
- after upload, verify selected file name or input state
- after download, verify saved file path and size
- replace `ok: true` only responses with verified outcome responses where possible

Done when:
- state-changing actions report verified results or explicit uncertainty
- false success rates are reduced

Verification:
- unit tests around fill, type, click, upload, and download verification behavior
- live MCP integration checks for at least one verified state-changing path

### Phase 4: Tighten Selector Strategy and Ambiguity Handling
Goal:
Reduce wrong actions caused by loose selector resolution.

Problem being solved:
- fragile automation on changing UI
- ambiguous text targeting

Primary files:
- [nl-command-parser.js](nl-command-parser.js)
- [edge-session.js](edge-session.js)
- [.github/skills/web-works/SKILL.md](.github/skills/web-works/SKILL.md)

Tasks:
- restrict risky action fallback to plain text matching
- prefer role, label, stable attributes, and skill-defined selectors for state-changing actions
- detect ambiguous matches and fail clearly instead of choosing one silently
- surface resolved selector strategy in action output

Done when:
- risky actions cannot silently act on ambiguous text matches
- selector resolution is visible and explainable in logs or tool output

Verification:
- add tests for ambiguous text targets
- add tests where multiple matching elements exist and the action must fail or require refinement

### Phase 5: Contain DOM Fallback and DOM Mutation Risk
Goal:
Keep fallback power without letting it become the main source of wrong actions.

Problem being solved:
- DOM fallback can bypass safer interaction behavior
- add and delete are powerful but risky

Primary files:
- [edge-session.js](edge-session.js)
- ~~agent-logic.js~~ (removed in v6.0)

Tasks:
- treat DOM fallback as recovery-only, not normal execution
- require tighter verification after DOM fallback succeeds
- limit or remove unsafe `innerHTML` injection paths from active flows
- review whether DOM delete/add should exist in the general-purpose interface at all
- keep safe defaults strict, even if local-first lowers platform risk

Done when:
- DOM fallback no longer masks ambiguous failures
- risky DOM mutation paths are gated, deprecated, or clearly limited

Verification:
- tests proving DOM fallback only runs after normal locator failure
- tests proving mutation actions require the intended policy state

### Phase 6: Add Secret and Data Hygiene
Goal:
Keep local-first power without leaking secrets into logs, traces, saved artifacts, or model-visible transcripts.

Problem being solved:
- secret exposure in local runtime outputs
- secret exposure in agent-visible tool transcripts

Primary files:
- [agent/agent/agent.py](agent/agent/agent.py)
- [agent/agent/runtime_utils.py](agent/agent/runtime_utils.py)
- [edge-session.js](edge-session.js)
- [cli-agent.js](cli-agent.js)
- [agent/agent/mcp_tools.py](agent/agent/mcp_tools.py)
- [agent/agent/tests/integration/test_mcp_live_integration.py](agent/agent/tests/integration/test_mcp_live_integration.py)
- [docs/PRODUCT_ARCHITECTURE.md](docs/PRODUCT_ARCHITECTURE.md)

Tasks:
- centralize redaction rules for keys, cookies, auth headers, and secret-bearing URLs
- sanitize tool arguments and tool results before printing them or appending them to model-visible transcripts
- sanitize browser snapshots, tab state, and saved evidence before failure artifacts are written
- keep raw copied trace and output artifacts opt-in when developers explicitly need them for debugging
- document the hygiene rules clearly for both runtimes

Done when:
- local execution remains powerful
- browser and agent logs no longer expose secrets by default
- failure artifacts keep useful debugging evidence without copying raw secret-bearing outputs unless explicitly requested

Verification:
- unit tests around redaction helpers and sanitized action responses
- agent-level review that tool-call logging and tool transcripts redact secrets
- live MCP failure-artifact review with raw output copying disabled by default

### Phase 7: Improve Evidence and Failure Artifacts
Goal:
Make failures debuggable and successes auditable.

Problem being solved:
- weak trust after failures
- unclear evidence after success

Primary files:
- [edge-session.js](edge-session.js)
- [agent/agent/mcp_tools.py](agent/agent/mcp_tools.py)
- [agent/agent/tests/integration/test_mcp_live_integration.py](agent/agent/tests/integration/test_mcp_live_integration.py)
- [logs/health](logs/health)

Tasks:
- standardize screenshots, snapshots, trace capture, and action evidence
- attach useful evidence to both success and failure paths where practical
- ensure local logs are understandable and not overloaded with noise

Done when:
- contributors can debug a browser failure from saved evidence without replaying blindly
- successful actions expose enough proof to build trust

Verification:
- live integration runs with trace and artifact review
- sample failure drill to confirm artifact completeness

### Phase 8: Unify Skill Contract and Runtime Selection
Goal:
Make the system predictable at the product level, not only the code level.

Problem being solved:
- unclear product behavior
- mismatch between VS Code and Python experiences

Primary files:
- [docs/PRODUCT_ARCHITECTURE.md](docs/PRODUCT_ARCHITECTURE.md)
- [.github/skills/web-works/SKILL.md](.github/skills/web-works/SKILL.md)
- [.github/skills/web-works/web-task.schema.json](.github/skills/web-works/web-task.schema.json)
- [agent/agent/workflow_tools.py](agent/agent/workflow_tools.py)

Tasks:
- define one shared workflow contract that both runtimes understand
- make runtime selection and execution mode selection explicit
- define how a successful run becomes a reusable artifact

Done when:
- developers can describe a workflow once and understand how either runtime will handle it
- product behavior is understandable before execution starts

Verification:
- docs review
- skill and workflow examples
- at least one end-to-end flow definition that can be interpreted consistently

## Recommended Implementation Order
Work in this order unless a blocker forces a change:

1. Phase 0
2. Phase 1
3. Phase 2
4. Phase 3
5. Phase 4
6. Phase 5
7. Phase 6
8. Phase 7
9. Phase 8

Reason:
- first stabilize the baseline
- then eliminate the biggest unsafe surfaces
- then unify behavior
- then harden targeting and secret hygiene
- finally refine product-level workflow consistency

## What Not To Do
- do not add new browser commands before the core action contract is stable
- do not expand DOM mutation features before verification and secret hygiene are stronger
- do not let the legacy JS helper path remain easier to use than the safer path
- do not mark a phase complete based only on manual intuition

## Completion Definition
A phase is complete only when all of the following are true:
- code changes are merged or ready
- verification steps pass
- docs are updated
- the unsafe behavior for that phase is reduced in a measurable way
- the next phase can begin without reopening the same core problem

## Practical First Three Work Passes
If the team wants the highest-value start, do these first:

### Pass 1
~~Contain or deprecate agent-logic.js and playwright-edge-agent.js.~~ **DONE: removed in v6.0**

### Pass 2
Port Python-style verification output into the JS path in [edge-session.js](edge-session.js).

### Pass 3
Restrict ambiguous text-based state-changing actions and make selector resolution visible.

Those three passes will reduce the biggest wrong-action and trust risks fastest.