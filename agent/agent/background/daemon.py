"""
Background Agent Daemon — Always-running process that manages the task queue.

Architecture (how real production daemons work):

1. DETACHED PROCESS — The daemon runs independent of the terminal.
   On Windows: subprocess with CREATE_NO_WINDOW + DETACHED_PROCESS flags.
   On Linux/Mac: double-fork or nohup to detach from terminal session.
   Closing the terminal does NOT kill the daemon.

2. HEARTBEAT — The daemon writes a timestamp every 5 seconds to daemon.json.
   Any process can check if the daemon is alive by reading the heartbeat age.

3. AUTO-RESTART (WATCHDOG) — A simple check that restarts the daemon if it dies.
   The --ensure command starts a daemon only if one isn't already running.

4. LOG ROTATION — Daemon logs to .agent-state/daemon.log with automatic rotation.
   Old logs are kept for 7 days, max 5MB per file.

5. GRACEFUL SHUTDOWN — SIGTERM/SIGINT triggers orderly shutdown.
   Active tasks finish before the daemon exits.

How the user interacts:
- Start daemon (background):  python -m agent.agent.background.daemon --start
- Start daemon (foreground):  python -m agent.agent.background.daemon --start --foreground
- Ensure daemon is running:   python -m agent.agent.background.daemon --ensure
- Stop daemon:                python -m agent.agent.background.daemon --stop
- Check daemon status:        python -m agent.agent.background.daemon --status
- Submit a task:              python -m agent.agent.background.daemon --submit "Fix the bug"
- Watch task progress:        python -m agent.agent.background.daemon --watch <task_id>
- List tasks:                 python -m agent.agent.background.daemon --list
- Get task result:            python -m agent.agent.background.daemon --result <task_id>
- Cancel task:                python -m agent.agent.background.daemon --cancel <task_id>
"""
import asyncio
import json
import logging
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from background.task_queue import TaskQueue, TaskStatus, TaskPriority
    from background.worker import execute_task
except Exception:
    from .task_queue import TaskQueue, TaskStatus, TaskPriority
    from .worker import execute_task


HEARTBEAT_INTERVAL = 5  # seconds
HEARTBEAT_MAX_AGE = 30  # seconds before considering daemon dead


class AgentDaemon:
    """Always-running background agent process with heartbeat and auto-restart."""

    def __init__(self, state_dir: Path = None, max_concurrent: int = 3, poll_interval: float = 2.0):
        if state_dir:
            self.state_dir = Path(state_dir).resolve()
        else:
            self.state_dir = Path(__file__).resolve().parents[3] / ".agent-state"
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.queue = TaskQueue(state_dir=self.state_dir)
        self.max_concurrent = max(1, min(int(max_concurrent), 10))
        self.poll_interval = max(0.5, min(float(poll_interval), 30.0))
        self.running = False
        self.active_tasks = {}  # task_id -> asyncio.Task
        self.daemon_file = self.state_dir / "daemon.json"
        self.pid_file = self.state_dir / "daemon.pid"
        self.log_file = self.state_dir / "daemon.log"
        self.logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        """Set up daemon logger with file rotation."""
        logger = logging.getLogger("agent-daemon")
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = RotatingFileHandler(
                str(self.log_file),
                maxBytes=5 * 1024 * 1024,  # 5MB
                backupCount=5,
                encoding="utf-8",
            )
            handler.setFormatter(logging.Formatter(
                "%(asctime)s [%(levelname)s] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            ))
            logger.addHandler(handler)
            # Also log to stderr if running in foreground
            if sys.stderr.isatty():
                console = logging.StreamHandler(sys.stderr)
                console.setFormatter(logging.Formatter("[Daemon] %(message)s"))
                logger.addHandler(console)
        return logger

    def _write_status(self, status: str):
        payload = {
            "status": status,
            "pid": os.getpid(),
            "started_at": getattr(self, "_started_at", None),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "heartbeat": time.time(),
            "active_tasks": list(self.active_tasks.keys()),
            "max_concurrent": self.max_concurrent,
            "poll_interval": self.poll_interval,
        }
        try:
            self.daemon_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception:
            pass

    def _write_pid(self):
        self.pid_file.write_text(str(os.getpid()), encoding="utf-8")

    def _remove_pid(self):
        try:
            if self.pid_file.exists():
                stored_pid = int(self.pid_file.read_text().strip())
                if stored_pid == os.getpid():
                    self.pid_file.unlink()
        except Exception:
            pass

    def _is_daemon_running(self) -> bool:
        """Check if a daemon is alive using PID + heartbeat."""
        if not self.pid_file.exists():
            return False
        try:
            pid = int(self.pid_file.read_text().strip())
            if pid == os.getpid():
                return True
            # Check if process is alive
            os.kill(pid, 0)
            # Process exists — also check heartbeat age
            if self.daemon_file.exists():
                status = json.loads(self.daemon_file.read_text(encoding="utf-8"))
                heartbeat = float(status.get("heartbeat", 0))
                if time.time() - heartbeat > HEARTBEAT_MAX_AGE:
                    self.logger.warning(f"Daemon PID {pid} exists but heartbeat stale ({time.time() - heartbeat:.0f}s)")
                    return False
            return True
        except (ProcessLookupError, OSError, ValueError):
            try:
                self.pid_file.unlink()
            except Exception:
                pass
            return False

    def get_status(self) -> dict:
        """Read current daemon status with liveness check."""
        alive = self._is_daemon_running()
        if self.daemon_file.exists():
            try:
                status = json.loads(self.daemon_file.read_text(encoding="utf-8"))
                heartbeat = float(status.get("heartbeat", 0))
                status["daemon_alive"] = alive
                status["heartbeat_age_seconds"] = round(time.time() - heartbeat, 1) if heartbeat else None
                return status
            except Exception:
                pass
        return {"status": "stopped", "daemon_alive": False}

    async def _run_task(self, task: dict):
        """Execute a task in an async context."""
        task_id = task["task_id"]
        self.logger.info(f"Starting task {task_id}: {task['prompt'][:100]}")
        try:
            await execute_task(task, self.queue, use_mcp=False)
            self.logger.info(f"Completed task {task_id}")
        except Exception as e:
            self.queue.fail(task_id, f"Daemon worker error: {e}")
            self.logger.error(f"Task {task_id} failed: {e}")
        finally:
            self.active_tasks.pop(task_id, None)
            self._write_status("running")

    async def _poll_loop(self):
        """Main daemon loop: poll queue, spawn workers, heartbeat, manage lifecycle."""
        self._started_at = datetime.now().isoformat(timespec="seconds")
        self.running = True
        self._write_pid()
        self._write_status("running")

        # Recover any stale tasks from previous crashes
        recovered = self.queue.recover_stale_tasks()
        if recovered:
            self.logger.info(f"Recovered {recovered} stale task(s) from previous crash")

        self.logger.info(f"Daemon started (PID={os.getpid()}, max_concurrent={self.max_concurrent})")

        cleanup_counter = 0
        heartbeat_counter = 0
        while self.running:
            try:
                # Clean up finished async tasks
                finished = [tid for tid, t in self.active_tasks.items() if t.done()]
                for tid in finished:
                    self.active_tasks.pop(tid, None)

                # Spawn workers for pending tasks if we have capacity
                while len(self.active_tasks) < self.max_concurrent:
                    task = self.queue.claim_next(worker_pid=os.getpid())
                    if not task:
                        break
                    task_id = task["task_id"]
                    self.logger.info(f"Claimed task {task_id}: {task['prompt'][:80]}")
                    async_task = asyncio.create_task(self._run_task(task))
                    self.active_tasks[task_id] = async_task

                # Heartbeat (every ~5 polls = ~10 seconds)
                heartbeat_counter += 1
                if heartbeat_counter >= max(1, int(HEARTBEAT_INTERVAL / self.poll_interval)):
                    heartbeat_counter = 0
                    self._write_status("running")

                # Periodic cleanup (every ~120 seconds)
                cleanup_counter += 1
                if cleanup_counter >= max(1, int(120 / self.poll_interval)):
                    cleanup_counter = 0
                    removed = self.queue.cleanup_old_tasks()
                    if removed:
                        self.logger.info(f"Cleaned up {removed} old task(s)")

                await asyncio.sleep(self.poll_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Poll error: {e}")
                await asyncio.sleep(self.poll_interval * 2)

        # Graceful shutdown: wait for active tasks
        if self.active_tasks:
            self.logger.info(f"Waiting for {len(self.active_tasks)} active task(s) to finish...")
            await asyncio.gather(*self.active_tasks.values(), return_exceptions=True)

        self._write_status("stopped")
        self._remove_pid()
        self.logger.info("Daemon stopped")

    def _handle_signal(self, signum, frame):
        self.logger.info(f"Received signal {signum}, shutting down...")
        self.running = False

    async def start_foreground(self):
        """Start the daemon in the foreground (blocks until stopped)."""
        if self._is_daemon_running():
            print("[Daemon] Another daemon is already running.")
            return False

        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

        await self._poll_loop()
        return True

    @staticmethod
    def start_detached():
        """Launch the daemon as a detached background process.
        This is the key method that makes it 'always running' —
        the process survives terminal close."""
        python = sys.executable
        daemon_module = "agent.agent.background.daemon"
        workspace_root = str(Path(__file__).resolve().parents[3])

        if sys.platform == "win32":
            # Windows: CREATE_NO_WINDOW + DETACHED_PROCESS
            CREATE_NO_WINDOW = 0x08000000
            DETACHED_PROCESS = 0x00000008
            proc = subprocess.Popen(
                [python, "-m", daemon_module, "--start", "--foreground"],
                cwd=workspace_root,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL,
                creationflags=CREATE_NO_WINDOW | DETACHED_PROCESS,
                close_fds=True,
            )
        else:
            # Unix: start_new_session detaches from terminal
            proc = subprocess.Popen(
                [python, "-m", daemon_module, "--start", "--foreground"],
                cwd=workspace_root,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
                close_fds=True,
            )

        print(f"[Daemon] Started in background (PID={proc.pid})")
        print(f"[Daemon] Logs: .agent-state/daemon.log")
        print(f"[Daemon] Status: python -m {daemon_module} --status")
        return proc.pid

    def stop(self):
        """Stop a running daemon by sending SIGTERM (Unix) or TerminateProcess (Windows)."""
        if not self.pid_file.exists():
            print("[Daemon] No daemon PID file found.")
            return False
        try:
            pid = int(self.pid_file.read_text().strip())
            if sys.platform == "win32":
                # Windows: use taskkill for clean shutdown
                subprocess.run(["taskkill", "/PID", str(pid), "/F"], capture_output=True)
            else:
                os.kill(pid, signal.SIGTERM)
            print(f"[Daemon] Stopped daemon (PID={pid})")
            self._remove_pid()
            self._write_status("stopped")
            return True
        except ProcessLookupError:
            print("[Daemon] Daemon process not found. Cleaning up stale files.")
            self._remove_pid()
            self._write_status("stopped")
            return False
        except Exception as e:
            print(f"[Daemon] Error stopping daemon: {e}")
            return False

    def ensure_running(self):
        """Start the daemon only if it's not already running. Safe to call repeatedly."""
        if self._is_daemon_running():
            status = self.get_status()
            print(f"[Daemon] Already running (PID={status.get('pid')}, heartbeat_age={status.get('heartbeat_age_seconds')}s)")
            return True
        print("[Daemon] Not running. Starting...")
        self.start_detached()
        # Wait briefly and verify it started
        time.sleep(2)
        if self._is_daemon_running():
            print("[Daemon] Started successfully.")
            return True
        print("[Daemon] Failed to start. Check .agent-state/daemon.log")
        return False


def _print_json(data):
    print(json.dumps(data, indent=2, ensure_ascii=False))


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Background Agent Daemon")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--start", action="store_true", help="Start the daemon (detached by default)")
    group.add_argument("--stop", action="store_true", help="Stop the daemon")
    group.add_argument("--status", action="store_true", help="Show daemon status")
    group.add_argument("--ensure", action="store_true", help="Start daemon only if not already running")
    group.add_argument("--submit", type=str, help="Submit a task")
    group.add_argument("--list", action="store_true", help="List tasks")
    group.add_argument("--result", type=str, help="Get task result by ID")
    group.add_argument("--cancel", type=str, help="Cancel a task by ID")
    group.add_argument("--watch", type=str, help="Watch a task's progress in real-time")
    parser.add_argument("--foreground", action="store_true", help="Run daemon in foreground (don't detach)")
    parser.add_argument("--priority", type=int, default=5, help="Task priority (0-10)")
    parser.add_argument("--max-concurrent", type=int, default=3, help="Max concurrent tasks")
    parser.add_argument("--filter", type=str, help="Filter tasks by status")
    args = parser.parse_args()

    daemon = AgentDaemon(max_concurrent=args.max_concurrent)

    if args.start:
        if args.foreground:
            asyncio.run(daemon.start_foreground())
        else:
            AgentDaemon.start_detached()

    elif args.stop:
        daemon.stop()

    elif args.status:
        _print_json(daemon.get_status())

    elif args.ensure:
        daemon.ensure_running()

    elif args.submit:
        task = daemon.queue.submit(args.submit, priority=args.priority)
        _print_json({"submitted": True, "task_id": task["task_id"], "status": task["status"]})

    elif args.list:
        tasks = daemon.queue.list_tasks(status=args.filter)
        _print_json({"count": len(tasks), "tasks": tasks})

    elif args.result:
        task = daemon.queue.get(args.result)
        if task:
            _print_json({
                "task_id": task["task_id"],
                "status": task["status"],
                "result": task.get("result"),
                "error": task.get("error"),
                "progress_count": len(task.get("progress", [])),
                "tool_calls": task.get("tool_calls", []),
            })
        else:
            _print_json({"error": f"Task not found: {args.result}"})

    elif args.cancel:
        result = daemon.queue.cancel(args.cancel)
        _print_json(result)

    elif args.watch:
        task_id = args.watch
        print(f"Watching task {task_id}... (Ctrl+C to stop)")
        seen = 0
        try:
            while True:
                task = daemon.queue.get(task_id)
                if not task:
                    print(f"Task not found: {task_id}")
                    break
                progress = task.get("progress", [])
                for entry in progress[seen:]:
                    ts = entry.get("timestamp", "")
                    msg = entry.get("message", "")
                    tool = entry.get("tool_name", "")
                    prefix = f"[{ts}]" if ts else ""
                    tool_prefix = f" ({tool})" if tool else ""
                    print(f"  {prefix}{tool_prefix} {msg}")
                seen = len(progress)
                if task["status"] in {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED}:
                    print(f"\nTask {task['status']}")
                    if task.get("result"):
                        print(f"Result: {task['result'][:2000]}")
                    if task.get("error"):
                        print(f"Error: {task['error']}")
                    break
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nStopped watching.")


if __name__ == "__main__":
    main()
