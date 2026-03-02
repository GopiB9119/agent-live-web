# web-works Prompt Library

Use these prompts with the `agent-live-web` agent.

## 1) Master Prompt (all-rounder)
```txt
Use playwright-edge MCP tools for interactive web steps and workspace tools for local file/code steps.
Work autonomously with strict safety and verification.

Execution profile: <balanced|deep|turbo>
Task ID: <task-id>
Checkpoint: .playwright-mcp/resume-state.json

Rules:
1) Understand first (website + files) before side effects.
2) One atomic action per step, then verify.
3) Retry once with improved strategy, then stop with blocker.
4) If confused/failing, switch to deep-research mode automatically.
5) Ask confirmation before send/submit/delete/purchase/merge/push.
6) Update checkpoint after every successful step.

Output per step:
Understanding:
Action:
Tool:
Verification:
Next:

Task:
<describe exact task>

Success criteria:
<measurable end conditions>
```

## 2) Resume Prompt (continue from previous chat)
```txt
Continue task_id=<task-id> from .playwright-mcp/resume-state.json.
Do not restart from step 1.

Resume preflight:
1) Select the real tab (not about:blank)
2) Verify URL and title
3) Capture snapshot
4) Re-resolve selectors from current DOM

Then continue from next step after last_completed_step_id.
Update checkpoint after each successful step.
```

## 3) Deep-Research Recovery Prompt
```txt
Switch to deep-research mode now.
Stop repeating the same failing action.

Collect evidence first:
1) URL/title + snapshot
2) Visible labels/roles/sections
3) Likely selector candidates (ordered by stability)
4) Top 2 execution strategies with risk

Pick safest strategy, execute one step, verify, continue.
```

## 4) Turbo Prompt (speed after clarity)
```txt
State is clear. Switch to turbo mode.
Minimize tool calls, keep strict verification, no redundant logs.
If any step fails once, do one improved retry, then blocker.
```

## 5) Website + Files Prompt
```txt
Execute website actions and local file/code updates in one run.
Order:
1) Understand website state
2) Complete browser actions with verification
3) Understand target files
4) Apply edits
5) Run validation commands
6) Return final evidence summary
```

