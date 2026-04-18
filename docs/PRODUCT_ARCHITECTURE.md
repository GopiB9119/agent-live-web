# Unified Product Architecture

## Objective
Define agent-live-web as one automation system with two runtimes and one shared workflow contract.

The intended product shape is:
- VS Code runtime as the default developer experience.
- Python runtime as an optional execution runtime.
- One skill-driven workflow model for both runtimes.
- Three execution modes per task: UI, API, and hybrid.

## Product Promise
Give the agent a workflow. The system chooses the right execution path, verifies each step, and produces reusable automation artifacts.

This is the shortest useful positioning for real developers:
- Describe the work once.
- Run it interactively in VS Code or headlessly in Python.
- Use browser automation when APIs are missing.
- Use APIs when they are faster and more stable.
- Export successful flows into reusable automation.

## Product Shape

### One System
The repo should be presented as one automation platform, not two separate products.

Shared system responsibilities:
- accept a user goal or reusable skill
- understand the target system
- choose UI, API, or hybrid execution
- enforce safety and side-effect controls
- verify outcomes with evidence
- save reusable outputs for developers

### Two Runtimes

#### 1. VS Code Runtime
Primary entry point for most developers.

Best for:
- interactive work
- process discovery
- supervised automation
- debugging real browser flows
- fast iteration while authoring a skill

Current repo mapping:
- MCP Edge runtime in [playwright-edge-mcp.js](playwright-edge-mcp.js)
- interactive browser control in [edge-session.js](edge-session.js)
- natural language CLI flow in [cli-agent.js](cli-agent.js)
- task skill support in [.github/skills/web-works/SKILL.md](.github/skills/web-works/SKILL.md)

#### 2. Python Runtime
Optional execution runtime for automation outside the editor.

Best for:
- CI jobs
- batch processing
- scheduled runs
- backend integrations
- programmatic orchestration

Current repo mapping:
- main runtime in [agent/agent/agent.py](agent/agent/agent.py)
- tool orchestration in [agent/agent/tools.py](agent/agent/tools.py)
- browser wrappers in [agent/agent/mcp_tools.py](agent/agent/mcp_tools.py)
- workflow planning in [agent/agent/workflow_tools.py](agent/agent/workflow_tools.py)

## Important Product Clarification
The phrase without API can mean two different things. The product should separate them clearly.

### Meaning A: Without model API setup
The user does not want to manage OpenAI or Azure keys.

Recommended experience:
- use VS Code runtime by default
- rely on the editor agent environment
- avoid requiring Python runtime unless the user explicitly wants it

### Meaning B: Without target-site backend API usage
The target website does not expose a useful API, or the user wants the real browser path.

Recommended experience:
- use UI automation through Playwright
- follow a skill or structured task definition
- verify against real page state, not assumptions

If these two meanings stay mixed together, the product story becomes confusing.

## Execution Modes

### UI Mode
Use Playwright against the real browser.

Use when:
- no backend API exists
- API access is blocked or unstable
- the goal depends on the real user interface
- the task requires visible end-user verification

Strengths:
- closest to real user behavior
- works on systems with weak API support
- strong trust signal when verified on screen

Risks:
- selectors can break
- flows can change with UI updates
- performance is slower than API execution

### API Mode
Use direct service or backend endpoints.

Use when:
- endpoint contracts are stable
- performance matters
- the task is data-heavy
- the action does not require UI validation

Strengths:
- faster
- more deterministic
- easier to run at scale

Risks:
- may diverge from real user behavior
- auth and secret handling become more important
- hidden side effects can be harder to explain to users

### Hybrid Mode
Use API for setup or bulk work and UI for verification or missing steps.

This should become the default advanced mode because it balances speed and trust.

Examples:
- use API to create test data, then verify the UI flow in the browser
- use API to fetch records, then use UI automation to complete the final manual-only step
- use API to authenticate or preload state, then run the browser flow to validate the actual user experience

## Shared Skill Contract
The skill should be the stable product contract across both runtimes.

Every reusable skill should define:
- objective
- start context
- allowed domains or target systems
- side-effect policy
- success criteria
- execution hints
- expected outputs
- resume state if the task spans multiple sessions

Natural-language requests can remain supported, but reusable flows should compile into a structured task format.

Current foundation already exists in:
- [.github/skills/web-works/SKILL.md](.github/skills/web-works/SKILL.md)
- [.github/skills/web-works/web-task.schema.json](.github/skills/web-works/web-task.schema.json)

## Developer Work Model
The main product behavior should be:
- understand the developer's codebase first
- accept the developer's work in a reusable skill or task format
- understand the target website deeply through Playwright
- choose the best execution path
- do the work
- write reusable scripts
- test those scripts
- explain the result back to the developer in a short and useful way

This means the system is not only a browser agent.
It is a codebase-aware developer workflow engine.

### Codebase Understanding
Before doing implementation work, the system should understand:
- relevant files in the repo
- what the developer is trying to change
- what existing scripts, tests, and runtime paths already exist
- what reusable artifact should be produced at the end
- what has already been completed in the current or previous run
- what the next recommended steps are if the work is resumed later

### Skill-Driven Input
The user should be able to provide work in a skill-like format that describes:
- the goal
- the steps or process
- the success criteria
- whether API access exists
- what final artifact should be produced

### Website Understanding
Before acting, the system should understand:
- what the website or app is doing
- where the relevant workflow lives
- which actions are safe
- whether the workflow is best done through UI, API, or hybrid execution

## Execution Paths for Developer Work

### Path A: Without API
Use this path when the developer does not want or cannot use backend APIs for the target system.

Expected behavior:
- VS Code agent is the default entry point
- Playwright is used to understand the real website flow in depth
- the agent executes the work from the user-provided skill
- the agent writes reusable browser automation scripts
- the agent tests those scripts
- the agent grounds its summary in verified execution results and saved artifacts
- the agent gives a brief summary of what happened and what the developer should know

This path is slower than direct API work, but it is closer to the real user experience and often necessary for web-only workflows.

### Path B: With API
Use this path when backend APIs exist and can accelerate the work.

Expected behavior:
- the agent still understands the website through Playwright first so the real flow is not misunderstood
- then the agent uses APIs where they make the task faster, more deterministic, or easier to scale
- the agent writes reusable scripts that can combine API actions with browser verification when needed
- the agent tests those scripts
- the agent grounds its summary in verified execution results and saved artifacts
- the agent gives a brief summary of what happened and what the developer should know

This path should usually be faster than UI-only automation, but it should not skip website understanding because that would reduce trust and product usefulness.

### Path C: Hybrid
Use this path when part of the work is faster through API and part of the work must still be validated in the real browser.

This is often the best long-term default because it combines:
- codebase understanding
- website understanding
- API speed
- UI trust and verification

## Proposed End-to-End Flow

### 1. User gives a task or skill
Input can be:
- natural language
- a structured web task JSON file
- a saved reusable skill

### 2. System understands the task
The system identifies:
- target domain
- side effects
- whether API access exists
- whether UI verification is required
- whether the run is interactive or headless
- what codebase area or reusable script target is involved

### 3. Runtime selection happens
Choose the runtime based on context:
- VS Code for interactive supervised work
- Python for automation outside the editor

### 4. Execution mode selection happens
Choose UI, API, or hybrid based on the task.

### 5. Execution runs with verification
Every meaningful step should include:
- plan
- execute
- verify
- recover once if needed

### 6. Reusable output is produced
Successful runs should produce one or more of:
- Playwright script
- structured workflow JSON
- Python job definition
- saved skill template
- run evidence and trace artifacts
- a short developer-facing explanation of what was done and why

## Why Developers Would Use This
Real-world developers usually want four things:
- low setup cost
- fast task completion
- reproducible automation
- clear control over risky actions

This product becomes attractive when it saves time on:
- repetitive internal admin tasks
- QA and regression checks
- onboarding workflows
- operational browser tasks
- hybrid UI plus API automation

## Local-First Risk Position
Because this system works locally, the overall platform risk is lower than a cloud-hosted autonomous system.

What local-first reduces:
- less concern about sending browser state to a remote service
- less infrastructure and multi-tenant exposure
- fewer deployment and hosting trust problems
- easier debugging because the developer owns the machine and session

That means the main product concern is not large-scale platform danger.
The main concern is whether the system is reliable, understandable, and controlled enough for developers to trust in daily work.

## Main Risks

### 1. Product confusion
If users cannot quickly understand why VS Code exists and when Python is needed, adoption will drop.

Required product rule:
- VS Code is the default experience
- Python is the optional runtime

### 2. Fragile browser automation
If flows break often and recovery is weak, trust collapses.

Required platform rule:
- verify every step
- capture evidence on failure
- keep retry behavior strict and explainable

### 3. Hidden setup cost
If first-run setup is too heavy, users will abandon the tool before value is visible.

Required product rule:
- deliver a useful first-run experience in VS Code with minimal configuration

### 4. No reusable output
If the agent finishes work but does not leave behind reusable artifacts, teams will not scale usage.

Required platform rule:
- convert successful runs into reusable assets

### 5. Unsafe autonomy
Even on a local machine, wrong actions can still cause real damage in the target app. The risk is smaller than a remote platform risk, but it is still a trust problem.

Required platform rule:
- auth material, cookies, secret-bearing URLs, and sensitive runtime artifacts must be redacted before they reach logs, traces, saved evidence, or model-visible transcripts

## Recommended Direction

### Near Term
- position the project publicly as one system with two runtimes
- keep VS Code as the default entry path
- keep Python clearly optional
- strengthen the shared skill contract
- make UI, API, and hybrid mode selection explicit in product language

### Medium Term
- add exportable automation artifacts after successful runs
- add runtime-selection logic based on task type and available capabilities
- add stronger API-mode orchestration where endpoints are known
- unify reporting, evidence, and trace artifacts across both runtimes

Implementation detail:
- use [docs/TRUST_RELIABILITY_EXECUTION_PLAN.md](docs/TRUST_RELIABILITY_EXECUTION_PLAN.md) as the phased delivery plan for browser trust, secret hygiene, and reliability work.

### Long Term
- let teams build libraries of reusable skills
- let successful interactive runs become headless jobs
- support mixed enterprise workflows where API and UI automation cooperate automatically

## Architecture Rule of Thumb
The runtime is an implementation detail.
The skill is the product contract.
The evidence is the trust layer.

If those three rules stay true, the platform can grow without becoming confusing.

## Success Criteria
This product direction is working when a developer can do all of the following with minimal friction:
- describe a workflow once
- run it in VS Code immediately
- reuse it later without manual re-teaching
- switch to Python only when headless or scheduled execution is needed
- understand why the system chose UI, API, or hybrid mode
- inspect evidence when something fails

## Practical Positioning Statement
agent-live-web is a skill-driven web automation platform for developers. It starts in VS Code, can scale into Python when needed, chooses UI, API, or hybrid execution based on the task, and turns successful runs into reusable automation.