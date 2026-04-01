# Social Media Launch Posts

Ready-to-use posts for launching **Agent Live Web** on LinkedIn, Product Hunt, Reddit, Twitter/X, Hacker News, and Dev.to.

---

## 🔗 LinkedIn

> **Tip:** Post as a personal update with the GitHub repo link and a short screen recording or screenshot.

---

**Option A – Professional / Storytelling**

```
🚀 I built a VS Code-first browser automation toolkit — and just open-sourced it.

Over the past few months I've been working on Agent Live Web — a Playwright Edge MCP toolkit
that lets you automate real websites directly inside VS Code, with tracing, evaluation,
and a local-first security model built in from day one.

Here's what it does:
✅ One-command MCP startup directly from the VS Code panel
✅ Step-by-step browser automation with built-in retry and verification logic
✅ OpenTelemetry tracing + Jaeger UI for debugging failed runs
✅ Python AI agent with OpenAI / Azure model switching — no code edits required
✅ Local evaluation pipeline to measure stability across releases
✅ OAuth-protected API support out of the box
✅ Memory + workspace management for long-running agent sessions
✅ No cloud required — browser profile and runtime stay fully local

It's MIT-licensed, v5.1.0 is stable, and the project is actively evolving toward v5.2
with full live MCP integration tests.

If you're building AI agents, browser automation pipelines, or VS Code tooling —
I'd love your feedback.

👉 GitHub: https://github.com/GopiB9119/agent-live-web

#OpenSource #BrowserAutomation #AIAgents #VSCode #Playwright #MCP #DevTools #Python
```

---

**Option B – Short / Punchy**

```
Just open-sourced Agent Live Web 🎉

A VS Code-first Playwright Edge MCP toolkit for:
→ Real-time browser automation with step verification
→ OpenTelemetry tracing & Jaeger UI debugging
→ Python AI agent (OpenAI + Azure) with model switching
→ Local-first security — no cloud, no data leaks
→ Built-in evaluation pipeline for stability tracking

MIT licensed. v5.1.0 is stable and ready to use.

🔗 https://github.com/GopiB9119/agent-live-web

Feedback welcome! 🙌

#VSCode #Playwright #AIAgents #OpenSource #BrowserAutomation #MCP
```

---

## 🐱 Product Hunt

> **How to post:** Go to https://www.producthunt.com/posts/new
> Fill in the fields below exactly.

**Name:** Agent Live Web

**Tagline:**
```
VS Code-first Playwright Edge MCP toolkit for real-time web automation
```

**Description:**
```
Agent Live Web is an open-source toolkit that brings reliable browser automation directly
into VS Code using Playwright Edge and the Model Context Protocol (MCP).

Key features:
• One-click MCP startup from the VS Code panel
• Real-website automation with step verification and retry controls
• OpenTelemetry tracing + Jaeger UI for fast failure debugging
• Python AI agent supporting OpenAI and Azure models — switchable via env vars
• OAuth-protected API support
• Local evaluation pipeline to measure and gate release stability
• Memory and workspace management for long-running agent sessions
• 100% local-first — browser profile and runtime never leave your machine

Perfect for developers building AI agents, QA automation pipelines, or VS Code extensions
that need trustworthy, auditable browser control.

MIT licensed. v5.1.0 stable.
```

**Website / GitHub:**
```
https://github.com/GopiB9119/agent-live-web
```

**Topics:** Developer Tools, Artificial Intelligence, Open Source, Productivity, Browser

**First Comment (Maker comment):**
```
Hey Product Hunt! 👋

I'm the maker of Agent Live Web. I built this because I kept running into the same
problems when building AI-powered browser automations:
- Flaky sessions with no visibility into what failed
- Secrets leaking through cloud automation providers
- No repeatable way to measure stability across code changes

Agent Live Web solves all three. Everything runs locally, every step is traced,
and the eval pipeline gives you a pass/fail gate before you ship.

Happy to answer any questions — fire away! 🚀
```

---

## 👾 Reddit

### r/programming

**Title:**
```
I open-sourced a VS Code-first Playwright Edge MCP toolkit for reliable browser automation
```

**Body:**
```
Hey r/programming,

I've been building Agent Live Web for the past few months and just released v5.1.0
as stable. It's an open-source toolkit that integrates Playwright Edge browser automation
directly into VS Code using the Model Context Protocol (MCP).

**What it does:**
- Start browser automation sessions directly from the VS Code MCP panel
- Execute real-website workflows with step-by-step verification and automatic retries
- Trace every action with OpenTelemetry + visualize in Jaeger UI
- Run a Python AI agent that supports both OpenAI and Azure models (switchable via env vars)
- Use an evaluation pipeline to gate releases on stability metrics
- Manage OAuth-protected APIs with a built-in profile system
- Everything stays local — no cloud calls, no data leaks

**Why I built it:**
Most browser automation tools either have poor observability, require cloud infrastructure,
or don't integrate well with VS Code's new AI/agent ecosystem. I wanted something
that was local-first, traceable, and had a built-in quality gate before shipping changes.

**Tech stack:** Node.js, Python, Playwright, OpenTelemetry, Jaeger, MCP

MIT licensed: https://github.com/GopiB9119/agent-live-web

Happy to answer questions or hear suggestions!
```

---

### r/vscode

**Title:**
```
I built a VS Code MCP extension for Playwright Edge browser automation — open source, v5.1.0 stable
```

**Body:**
```
Hey r/vscode!

Just released v5.1.0 of Agent Live Web — a VS Code-first Playwright Edge MCP toolkit
for automating real websites directly from the VS Code MCP panel.

**VS Code integration highlights:**
- Start/stop the MCP session from the VS Code panel (no terminal needed)
- Owner lock system prevents conflicts when multiple sessions try to use the same Edge profile
- VS Code Copilot-compatible task instructions via `.github/copilot-instructions.md`

**Other features:**
- Step verification + retry controls on every browser action
- OpenTelemetry traces → Jaeger UI for debugging
- Python AI agent with OpenAI / Azure model switching
- Evaluation pipeline for release gating
- Local-first: profile and runtime never leave your machine

GitHub: https://github.com/GopiB9119/agent-live-web

Would love to hear from other VS Code extension / MCP tool builders!
```

---

### r/selfhosted

**Title:**
```
Self-hosted browser automation for VS Code with full local tracing — Agent Live Web v5.1.0
```

**Body:**
```
Hey r/selfhosted,

If you want browser automation that stays 100% on your machine, check out
Agent Live Web — a VS Code-first Playwright Edge MCP toolkit I just released.

**Why it fits self-hosted:**
- Browser profile and MCP runtime are fully local
- No cloud dependencies — everything runs on your own hardware
- Tracing via OpenTelemetry → self-hosted Jaeger UI (Docker Compose included)
- Python AI agent calls your own OpenAI-compatible or Azure endpoint
- Sensitive files excluded from git by default

MIT licensed: https://github.com/GopiB9119/agent-live-web

Happy to answer questions about self-hosting the tracing stack!
```

---

### r/MachineLearning / r/LocalLLaMA

**Title:**
```
Open-source VS Code agent toolkit for real browser automation — supports OpenAI + Azure, local-first
```

**Body:**
```
Hey all,

Sharing Agent Live Web — an open-source toolkit I built for running AI agents
that control a real browser directly from VS Code.

**AI/LLM-relevant features:**
- Python agent supports OpenAI and Azure providers, switchable via environment variables
- Model name is fully configurable — drop in any compatible future model
- OAuth-protected API integration for agents that need to call external services
- Memory and workspace management for long-running multi-turn agent sessions
- Evaluation pipeline to measure agent response quality over time
- No cloud required — all execution stays local

The goal was to give AI agents reliable, auditable browser control with minimal
configuration overhead.

MIT licensed: https://github.com/GopiB9119/agent-live-web

Would be curious what other people are using for local browser-control agents!
```

---

## 🐦 Twitter / X

**Thread (recommended):**

```
Tweet 1:
🚀 Just open-sourced Agent Live Web v5.1.0 — a VS Code-first Playwright Edge MCP toolkit
for reliable browser automation with tracing + eval built in.

MIT licensed 🔓 → https://github.com/GopiB9119/agent-live-web

🧵 Here's what it does:

Tweet 2:
🖥️ Runs entirely inside VS Code.
Start/stop browser sessions from the MCP panel — no terminal juggling.
Owner lock prevents profile conflicts across multiple agents.

Tweet 3:
🔍 Every action is traced.
OpenTelemetry → Jaeger UI.
See exactly what ran, what failed, and why. Fast triage scripts included.

Tweet 4:
🤖 Python AI agent included.
Switch between OpenAI + Azure with a single env var.
No code edits. Drop in any compatible model name.

Tweet 5:
🔐 Local-first security.
Browser profile stays on your machine.
Sensitive files excluded from git. Side-effect actions require confirmation.

Tweet 6:
📊 Built-in evaluation pipeline.
Run eval:gate before shipping any change.
Pass/fail quality gate that tracks stability over time.

Tweet 7:
👉 Get started:
npm install && npm run install:edge && npm run mcp:edge

GitHub: https://github.com/GopiB9119/agent-live-web

Feedback welcome! 🙌 #OpenSource #AIAgents #VSCode #Playwright #BrowserAutomation #MCP
```

**Single tweet (short version):**

```
🚀 Just open-sourced Agent Live Web v5.1.0

VS Code + Playwright Edge + MCP = reliable browser automation with:
→ OpenTelemetry tracing
→ Python AI agent (OpenAI/Azure)
→ Local-first security
→ Built-in eval pipeline

MIT 🔓 → https://github.com/GopiB9119/agent-live-web

#OpenSource #AIAgents #VSCode
```

---

## 🟠 Hacker News (Show HN)

**Title:**
```
Show HN: Agent Live Web – VS Code Playwright Edge MCP toolkit with tracing and eval pipeline
```

**Body:**
```
I've been building Agent Live Web (https://github.com/GopiB9119/agent-live-web)
for the past several months and am sharing v5.1.0 today.

It's a VS Code-first toolkit that combines Playwright Edge browser control,
Model Context Protocol (MCP) session management, OpenTelemetry tracing,
and a Python AI agent into a single local-first package.

The motivation: I wanted browser automation that (1) had full observability,
(2) never required cloud infrastructure, and (3) had a repeatable quality gate
before deploying changes to agents.

Key design decisions I'd love feedback on:
- MCP owner lock: only one VS Code process can own the Edge profile at a time.
  This prevents silent race conditions between agents but adds friction for multi-agent setups.
- Eval pipeline: every release runs a pass/fail gate against a generated dataset.
  Right now the threshold is configurable (lenient/normal/strict profiles).
- Python agent model config: provider + model name are env vars, not code,
  so you can swap OpenAI ↔ Azure without touching source.

Stack: Node.js, Python 3.12, Playwright, OpenTelemetry SDK, Jaeger, MCP.
MIT licensed.

Would especially appreciate feedback from people building VS Code extensions,
local AI agent frameworks, or Playwright-based QA pipelines.
```

---

## ✍️ Dev.to

**Title:**
```
I built a VS Code-first browser automation toolkit with tracing, eval, and a Python AI agent — here's how it works
```

**Tags:** `#vscode #playwright #automation #opensource`

**Intro paragraph:**
```
If you've tried to build reliable browser automation for AI agents, you've probably run
into the same frustrations I did: flaky sessions, no visibility into what failed, and
no way to measure whether a code change made things better or worse.

Agent Live Web is my answer to those problems. It's an open-source VS Code-first toolkit
that brings Playwright Edge browser control, MCP session management, OpenTelemetry tracing,
and a Python AI agent together in a single local-first package.

In this post I'll walk through the core architecture and the decisions I made along the way.
```

> **Full article outline:**
> 1. What problem does it solve?
> 2. Architecture overview (VS Code MCP panel → Edge session → Python agent → Tracing → Eval)
> 3. Local-first security model (why it matters for AI agents)
> 4. The evaluation pipeline (how to gate releases on stability)
> 5. Getting started (3-command quickstart)
> 6. What's next (v5.2 live MCP integration tests)

---

## 📋 Quick Reference — Repo Facts

| Field | Value |
|---|---|
| GitHub URL | https://github.com/GopiB9119/agent-live-web |
| Current version | v5.1.0 |
| License | MIT |
| Primary language | Node.js + Python 3.12 |
| Key dependencies | Playwright, OpenTelemetry, Jaeger, MCP |
| VS Code integration | MCP panel (playwright-edge) |
| AI providers | OpenAI, Azure OpenAI |
| Description | VS Code-first Playwright Edge MCP toolkit for reliable real-time web automation, tracing, and evaluation |
| Topics | playwright, mcp, vscode, github-copilot, edge, browser-automation, web-automation, ai-agent, nodejs, opentelemetry, jaeger, tracing, observability, qa, automation |
