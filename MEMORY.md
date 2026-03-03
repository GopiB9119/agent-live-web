# Curated Long-Term Memory

## 2026-03-03 - Web/chat reliability hardening
- Root issues observed: overlay/state drift, stale selector targeting, weak chat verification, stale response extraction, and noisy diagnostics.
- Updated agent policy in `.github/agents/agent-live-web.agent.md` with mandatory hardening addendum (`H1`-`H7`).
- Propagated same controls to repository instruction layer:
	- `.github/copilot-instructions.md`
	- `.github/instructions/playwright-edge.instructions.md`
- Documented contributor-facing standards in `README.md` under `Reliability standards (web + chatbot)`.
- Operational rule now enforced: for chatbot steps, success requires both send confirmation and a new assistant response after the send.