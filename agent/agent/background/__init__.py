"""
Background Agent Daemon — Always-running agent process manager.

Architecture (inspired by Claude Code / Codex CLI / Devin):
┌──────────────────────────────────────────────────┐
│                  USER (frontend)                  │
│  CLI / VS Code / API / Web UI                     │
│  Sends tasks, reads results, streams progress     │
└──────────┬───────────────────────────┬───────────┘
           │ submit task               │ read results
┌──────────▼───────────────────────────▼───────────┐
│              TASK QUEUE (task_queue.py)            │
│  File-based persistent queue in .agent-state/     │
│  States: pending → running → completed/failed     │
└──────────┬───────────────────────────────────────┘
           │ pick up tasks
┌──────────▼───────────────────────────────────────┐
│          BACKGROUND DAEMON (daemon.py)            │
│  Always running, polls queue, executes tasks      │
│  Manages worker pool, handles graceful shutdown   │
│  Logs to .agent-state/daemon.log                  │
└──────────┬───────────────────────────────────────┘
           │ delegates to
┌──────────▼───────────────────────────────────────┐
│         WORKER AGENTS (worker.py)                 │
│  Each task gets its own agent loop                │
│  Uses same tools.py / AVAILABLE_FUNCTIONS         │
│  Streams results back to task queue               │
│  Can spawn sub-workers for parallel work          │
└──────────────────────────────────────────────────┘

Key design decisions:
- File-based queue for simplicity and crash recovery (no Redis/DB needed)
- Each task is an independent JSON file in .agent-state/tasks/
- Daemon is a single long-lived process that manages everything
- Workers share the same tool registry as the interactive agent
- Results stream to task file so frontend can poll or watch
"""
