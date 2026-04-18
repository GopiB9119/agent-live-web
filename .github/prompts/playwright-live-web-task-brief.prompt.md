---
name: "Playwright Live Web Task Brief"
description: "Use when starting a VS Code Copilot Playwright Edge MCP, Playwright MCP, or live web autonomous task. Produces a structured brief with what to do, what not to do, why, who, how, risks, bad impact, evidence, and completion criteria."
argument-hint: "Describe the website task, what to avoid, why it matters, risks, and required evidence."
agent: "agent-live-web"
---

Convert the user request into a structured execution brief for the `agent-live-web` custom agent.

Return exactly these sections:
- `What I Want:`
- `What I Do Not Want:`
- `Why:`
- `Surface:`
- `Who / Owner / Login Context:`
- `How To Work:`
- `Risks / Bad Impact To Avoid:`
- `Evidence Required:`
- `Stop And Ask Before:`
- `Done When:`
- `First Safe Step:`

Rules:
- Keep the brief grounded in the user request and current repo context.
- If a field is missing, add `Assumption:` with the safest reversible assumption.
- In `How To Work`, choose one task mode (`explore`, `extract`, or `automate`) and one execution profile (`balanced`, `deep`, or `turbo`).
- Put irreversible actions under `Stop And Ask Before`.
- Do not start execution. Produce the brief only.