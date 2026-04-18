"""
Background task tools — allow the interactive agent to dispatch work to the background daemon.

These tools let the foreground agent:
1. Submit tasks to the background queue
2. Check on background task status
3. Get results from completed background tasks
4. Cancel running background tasks
5. List all background tasks

This is how Claude Code's headless mode works: the user talks to the foreground agent,
which can delegate long-running work to background workers.
"""
import json
from pathlib import Path
try:
    from runtime_utils import redact_sensitive_data as _redact_sensitive_data
except Exception:
    from .runtime_utils import redact_sensitive_data as _redact_sensitive_data

try:
    from background.task_queue import TaskQueue, TaskPriority
except Exception:
    from .background.task_queue import TaskQueue, TaskPriority


class BackgroundTaskManager:
    """Manages background task submission and monitoring from the interactive agent."""

    def __init__(self, state_dir: Path = None):
        self.queue = TaskQueue(state_dir=state_dir)

    @staticmethod
    def _json_response(payload, max_chars=30000):
        return json.dumps(_redact_sensitive_data(payload, max_chars=max_chars), ensure_ascii=True)

    async def bg_submit(self, kwargs_dict):
        """Submit a task to run in the background.
        The background daemon will pick it up and execute it autonomously."""
        kwargs = kwargs_dict or {}
        prompt = str(kwargs.get("prompt", "")).strip()
        priority = int(kwargs.get("priority", TaskPriority.NORMAL))
        depends_on = kwargs.get("depends_on", [])
        max_iterations = int(kwargs.get("max_iterations", 10))

        if not prompt:
            return self._json_response({"status": "failed", "error": "prompt is required"})

        task = self.queue.submit(
            prompt=prompt,
            priority=priority,
            depends_on=depends_on if isinstance(depends_on, list) else [],
            max_iterations=max_iterations,
        )

        return self._json_response({
            "status": "ok",
            "task_id": task["task_id"],
            "priority": task["priority"],
            "message": f"Task submitted. The background daemon will execute it. Use bg_status with task_id='{task['task_id']}' to check progress.",
        })

    async def bg_status(self, kwargs_dict):
        """Check the status and progress of a background task."""
        kwargs = kwargs_dict or {}
        task_id = str(kwargs.get("task_id", "")).strip()

        if not task_id:
            return self._json_response({"status": "failed", "error": "task_id is required"})

        task = self.queue.get(task_id)
        if not task:
            return self._json_response({"status": "failed", "error": f"Task not found: {task_id}"})

        progress = task.get("progress", [])
        recent_progress = progress[-10:] if progress else []

        return self._json_response({
            "status": "ok",
            "task_id": task["task_id"],
            "task_status": task["status"],
            "created_at": task.get("created_at"),
            "started_at": task.get("started_at"),
            "completed_at": task.get("completed_at"),
            "progress_count": len(progress),
            "recent_progress": recent_progress,
            "has_result": bool(task.get("result")),
            "has_error": bool(task.get("error")),
        })

    async def bg_result(self, kwargs_dict):
        """Get the full result of a completed background task."""
        kwargs = kwargs_dict or {}
        task_id = str(kwargs.get("task_id", "")).strip()

        if not task_id:
            return self._json_response({"status": "failed", "error": "task_id is required"})

        task = self.queue.get(task_id)
        if not task:
            return self._json_response({"status": "failed", "error": f"Task not found: {task_id}"})

        return self._json_response({
            "status": "ok",
            "task_id": task["task_id"],
            "task_status": task["status"],
            "result": task.get("result"),
            "error": task.get("error"),
            "tool_calls": task.get("tool_calls", []),
            "progress_count": len(task.get("progress", [])),
        })

    async def bg_cancel(self, kwargs_dict):
        """Cancel a background task."""
        kwargs = kwargs_dict or {}
        task_id = str(kwargs.get("task_id", "")).strip()

        if not task_id:
            return self._json_response({"status": "failed", "error": "task_id is required"})

        result = self.queue.cancel(task_id)
        return self._json_response(result)

    async def bg_list(self, kwargs_dict=None):
        """List all background tasks."""
        kwargs = kwargs_dict or {}
        status_filter = str(kwargs.get("status", "")).strip() or None
        limit = max(1, min(int(kwargs.get("limit", 20)), 100))

        tasks = self.queue.list_tasks(status=status_filter, limit=limit)
        return self._json_response({
            "status": "ok",
            "count": len(tasks),
            "tasks": tasks,
        })
