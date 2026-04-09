"""
Task Queue — File-based persistent task management.

Each task is a JSON file in .agent-state/tasks/{task_id}.json
Tasks flow through states: pending → running → completed | failed | cancelled

Supports:
- Task submission with priority
- Task cancellation
- Task progress streaming
- Crash recovery (running tasks reset to pending on daemon restart)
- Task dependencies (task B waits for task A)
- Task history with retention
"""
import json
import os
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

try:
    from runtime_utils import redact_sensitive_data as _redact_sensitive_data
except Exception:
    from ..runtime_utils import redact_sensitive_data as _redact_sensitive_data


class TaskStatus:
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskPriority:
    LOW = 0
    NORMAL = 5
    HIGH = 8
    CRITICAL = 10


class TaskQueue:
    """File-based persistent task queue."""

    def __init__(self, state_dir: Optional[Path] = None):
        if state_dir:
            self.state_dir = Path(state_dir).resolve()
        else:
            self.state_dir = Path(__file__).resolve().parents[3] / ".agent-state"
        self.tasks_dir = self.state_dir / "tasks"
        self.tasks_dir.mkdir(parents=True, exist_ok=True)

    def _task_path(self, task_id: str) -> Path:
        safe_id = str(task_id).replace("/", "_").replace("\\", "_")
        return self.tasks_dir / f"{safe_id}.json"

    def _read_task(self, task_id: str) -> Optional[dict]:
        path = self._task_path(task_id)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def _write_task(self, task: dict):
        task_id = task.get("task_id", "")
        if not task_id:
            return
        path = self._task_path(task_id)
        task["updated_at"] = datetime.now().isoformat(timespec="seconds")
        path.write_text(json.dumps(task, indent=2, ensure_ascii=True), encoding="utf-8")

    def submit(
        self,
        prompt: str,
        priority: int = TaskPriority.NORMAL,
        depends_on: Optional[list] = None,
        metadata: Optional[dict] = None,
        max_iterations: int = 10,
        tools_allowed: Optional[list] = None,
    ) -> dict:
        """Submit a new task to the queue. Returns the task object."""
        task_id = f"task-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
        task = {
            "task_id": task_id,
            "status": TaskStatus.PENDING,
            "priority": max(0, min(int(priority), 10)),
            "prompt": str(prompt).strip(),
            "depends_on": depends_on or [],
            "metadata": metadata or {},
            "max_iterations": max(1, min(int(max_iterations), 40)),
            "tools_allowed": tools_allowed,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "started_at": None,
            "completed_at": None,
            "worker_pid": None,
            "progress": [],
            "result": None,
            "error": None,
            "tool_calls": [],
            "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }
        self._write_task(task)
        return task

    def get(self, task_id: str) -> Optional[dict]:
        """Get a task by ID."""
        return self._read_task(task_id)

    def cancel(self, task_id: str) -> dict:
        """Cancel a pending or running task."""
        task = self._read_task(task_id)
        if not task:
            return {"status": "failed", "error": f"Task not found: {task_id}"}
        if task["status"] in {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED}:
            return {"status": "failed", "error": f"Task already in terminal state: {task['status']}"}
        task["status"] = TaskStatus.CANCELLED
        task["completed_at"] = datetime.now().isoformat(timespec="seconds")
        self._write_task(task)
        return {"status": "ok", "task_id": task_id, "cancelled": True}

    def claim_next(self, worker_pid: int) -> Optional[dict]:
        """Claim the next available pending task (highest priority first).
        Uses atomic file-locking to prevent race conditions between workers.
        Returns the task if claimed, None if queue is empty."""
        pending = []
        for task_file in self.tasks_dir.glob("task-*.json"):
            try:
                task = json.loads(task_file.read_text(encoding="utf-8"))
                if task.get("status") != TaskStatus.PENDING:
                    continue
                # Check dependencies
                deps = task.get("depends_on", [])
                all_deps_done = True
                for dep_id in deps:
                    dep = self._read_task(dep_id)
                    if not dep or dep.get("status") != TaskStatus.COMPLETED:
                        all_deps_done = False
                        break
                if all_deps_done:
                    pending.append(task)
            except Exception:
                continue

        if not pending:
            return None

        # Sort by priority (highest first), then by creation time (oldest first)
        pending.sort(key=lambda t: (-t.get("priority", 5), t.get("created_at", "")))

        # Atomic claim using lock file to prevent race condition
        for chosen in pending:
            task_id = chosen.get("task_id", "")
            lock_path = self._task_path(task_id).with_suffix(".lock")
            try:
                # Try to create lock file exclusively (atomic on most OS)
                fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.close(fd)
            except FileExistsError:
                # Another worker already claimed this task
                continue

            try:
                # Re-read to verify still pending (double-check after lock)
                fresh = self._read_task(task_id)
                if not fresh or fresh.get("status") != TaskStatus.PENDING:
                    lock_path.unlink(missing_ok=True)
                    continue

                fresh["status"] = TaskStatus.RUNNING
                fresh["started_at"] = datetime.now().isoformat(timespec="seconds")
                fresh["worker_pid"] = worker_pid
                self._write_task(fresh)
                return fresh
            finally:
                lock_path.unlink(missing_ok=True)

        return None

    def update_progress(self, task_id: str, message: str, tool_name: str = "", tool_result: str = ""):
        """Append a progress entry to a running task."""
        task = self._read_task(task_id)
        if not task:
            return
        entry = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "message": str(message)[:2000],
        }
        if tool_name:
            entry["tool_name"] = str(tool_name)
            entry["tool_result_preview"] = str(tool_result)[:500]
        task.setdefault("progress", []).append(entry)
        # Keep only last 100 progress entries
        task["progress"] = task["progress"][-100:]
        self._write_task(task)

    def complete(self, task_id: str, result: str, tool_calls: Optional[list] = None):
        """Mark a task as completed with its result."""
        task = self._read_task(task_id)
        if not task:
            return
        task["status"] = TaskStatus.COMPLETED
        task["completed_at"] = datetime.now().isoformat(timespec="seconds")
        task["result"] = _redact_sensitive_data(str(result)[:50000], max_chars=50000)
        if tool_calls:
            task["tool_calls"] = tool_calls[-50:]
        self._write_task(task)

    def fail(self, task_id: str, error: str):
        """Mark a task as failed."""
        task = self._read_task(task_id)
        if not task:
            return
        task["status"] = TaskStatus.FAILED
        task["completed_at"] = datetime.now().isoformat(timespec="seconds")
        task["error"] = str(error)[:5000]
        self._write_task(task)

    def list_tasks(self, status: Optional[str] = None, limit: int = 50) -> list:
        """List tasks, optionally filtered by status."""
        tasks = []
        for task_file in sorted(self.tasks_dir.glob("task-*.json"), key=lambda p: p.name, reverse=True):
            try:
                task = json.loads(task_file.read_text(encoding="utf-8"))
                if status and task.get("status") != status:
                    continue
                # Return summary without full progress/result
                tasks.append({
                    "task_id": task.get("task_id"),
                    "status": task.get("status"),
                    "priority": task.get("priority"),
                    "prompt": task.get("prompt", "")[:200],
                    "created_at": task.get("created_at"),
                    "started_at": task.get("started_at"),
                    "completed_at": task.get("completed_at"),
                    "progress_count": len(task.get("progress", [])),
                    "has_result": bool(task.get("result")),
                    "has_error": bool(task.get("error")),
                })
                if len(tasks) >= limit:
                    break
            except Exception:
                continue
        return tasks

    def recover_stale_tasks(self, max_age_seconds: int = 600):
        """Reset running tasks that appear abandoned (worker crashed).
        Called on daemon startup."""
        now = time.time()
        recovered = 0
        for task_file in self.tasks_dir.glob("task-*.json"):
            try:
                task = json.loads(task_file.read_text(encoding="utf-8"))
                if task.get("status") != TaskStatus.RUNNING:
                    continue
                started = task.get("started_at", "")
                if started:
                    started_ts = datetime.fromisoformat(started).timestamp()
                    if now - started_ts > max_age_seconds:
                        task["status"] = TaskStatus.PENDING
                        task["started_at"] = None
                        task["worker_pid"] = None
                        task["progress"].append({
                            "timestamp": datetime.now().isoformat(timespec="seconds"),
                            "message": "[RECOVERED] Task was stale and reset to pending.",
                        })
                        self._write_task(task)
                        recovered += 1
            except Exception:
                continue
        return recovered

    def cleanup_old_tasks(self, max_age_days: int = 7, keep_latest: int = 100):
        """Remove old completed/failed/cancelled tasks."""
        task_files = sorted(self.tasks_dir.glob("task-*.json"), key=lambda p: p.name, reverse=True)
        removed = 0
        cutoff = time.time() - (max_age_days * 86400)
        for task_file in task_files[keep_latest:]:
            try:
                task = json.loads(task_file.read_text(encoding="utf-8"))
                if task.get("status") in {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED}:
                    completed = task.get("completed_at", "")
                    if completed:
                        completed_ts = datetime.fromisoformat(completed).timestamp()
                        if completed_ts < cutoff:
                            task_file.unlink()
                            removed += 1
            except Exception:
                continue
        return removed
