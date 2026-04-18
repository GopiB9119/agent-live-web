import inspect
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
try:
    from runtime_utils import redact_sensitive_data as _redact_sensitive_data, redact_sensitive_text as _redact_sensitive_text
except Exception:
    from .runtime_utils import redact_sensitive_data as _redact_sensitive_data, redact_sensitive_text as _redact_sensitive_text


class WorkflowManager:
    """
    Planning and orchestration manager for autonomous multi-step workflows.
    """

    def __init__(
        self,
        available_functions_provider,
        is_probably_text_source_fn,
        codebase_analyze_fn,
        fs_analyze_file_fn,
    ):
        self.available_functions_provider = available_functions_provider
        self.is_probably_text_source = is_probably_text_source_fn
        self.codebase_analyze_fn = codebase_analyze_fn
        self.fs_analyze_file_fn = fs_analyze_file_fn

    @staticmethod
    def _sanitize_payload(value, max_chars=50000):
        return _redact_sensitive_data(value, max_chars=max_chars)

    @classmethod
    def _json_response(cls, payload, max_chars=50000):
        return json.dumps(cls._sanitize_payload(payload, max_chars=max_chars), ensure_ascii=True)

    @staticmethod
    def _timestamp_utc() -> str:
        return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

    @staticmethod
    def _short_text(value, max_chars: int = 220) -> str:
        text = _redact_sensitive_text(value, max_chars=max_chars)
        return text.strip()

    def _summarize_tool_result(self, tool_name: str, result: dict) -> str:
        payload = self._sanitize_payload(result or {}, max_chars=2000)
        verification = payload.get("verification") if isinstance(payload.get("verification"), dict) else {}
        if verification.get("reason"):
            return self._short_text(verification.get("reason"), max_chars=220)
        if payload.get("error"):
            return self._short_text(payload.get("error"), max_chars=220)
        parts = []
        for key in ["path", "url", "title", "count", "status_code", "selected_index", "executed_steps"]:
            value = payload.get(key)
            if value is None or value == "" or value == []:
                continue
            parts.append(f"{key}={value}")
            if len(parts) >= 3:
                break
        if parts:
            return "; ".join(parts)
        return f"{tool_name or 'tool'} completed."

    def _build_workflow_artifact(self, execution, overall_status: str, duration_ms: int, workflow_goal: str = ""):
        ok_steps = 0
        failed_entries = []
        step_summaries = []
        for entry in execution:
            result = entry.get("result") if isinstance(entry.get("result"), dict) else {}
            status = entry.get("status", result.get("status", "ok"))
            if status in {"ok", "success"}:
                ok_steps += 1
            else:
                failed_entries.append(entry)
            step_summaries.append(
                {
                    "step": entry.get("step"),
                    "tool_name": entry.get("tool_name", ""),
                    "required": bool(entry.get("required", True)),
                    "status": status,
                    "duration_ms": int(result.get("duration_ms", 0) or 0),
                    "summary": self._summarize_tool_result(entry.get("tool_name", ""), result),
                }
            )

        if failed_entries:
            first_failed = failed_entries[0]
            developer_summary = (
                f"Workflow failed after {ok_steps}/{len(execution)} successful steps. "
                f"First failure: step {first_failed.get('step')} ({first_failed.get('tool_name')}) - "
                f"{self._summarize_tool_result(first_failed.get('tool_name', ''), first_failed.get('result', {}))}"
            )
        else:
            developer_summary = f"Workflow completed successfully with {ok_steps}/{len(execution)} steps in {duration_ms}ms."

        next_actions = [
            "Reuse the step summaries to turn this run into a stable skill, script, or test flow.",
            "Rerun the single failing step directly if you need deeper debugging.",
            "Persist this sanitized artifact externally if you want a reusable execution record.",
        ]

        return {
            "kind": "workflow_execution",
            "format_version": 1,
            "generated_at": self._timestamp_utc(),
            "goal": workflow_goal,
            "status": overall_status,
            "executed_steps": len(execution),
            "duration_ms": duration_ms,
            "step_summaries": step_summaries,
            "developer_summary": developer_summary,
            "next_actions": next_actions,
        }

    def _build_task_artifact(self, objective: str, path_value, plan: dict, focus_files, deep_file_analysis, status: str):
        focus_paths = [str(path) for path in (focus_files or [])]
        developer_summary = (
            f"Autopilot analyzed '{objective}' and selected {len(focus_paths)} focus files for deeper work."
            if status == "ok"
            else f"Autopilot could not complete objective '{objective}'."
        )
        return {
            "kind": "task_autopilot",
            "format_version": 1,
            "generated_at": self._timestamp_utc(),
            "objective": objective,
            "path": path_value,
            "status": status,
            "plan_steps": plan.get("plan_steps", []) if isinstance(plan, dict) else [],
            "focus_files": focus_paths,
            "focus_file_count": len(focus_paths),
            "deep_analysis_count": len(deep_file_analysis or []),
            "developer_summary": developer_summary,
            "next_actions": [
                "Turn the focus files into a deterministic workflow_execute plan.",
                "Use fs_patch or fs_edit_lines for code changes, then run tests.",
                "Save or reuse this artifact as the starting point for follow-up work.",
            ],
        }

    @staticmethod
    def coerce_tool_result_to_dict(raw_result):
        if isinstance(raw_result, dict):
            return _redact_sensitive_data(raw_result, max_chars=50000)
        text = str(raw_result)
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return _redact_sensitive_data(parsed, max_chars=50000)
        except Exception:
            pass
        return {"status": "ok", "raw": _redact_sensitive_text(text, max_chars=50000)}

    def _get_available_functions(self):
        if callable(self.available_functions_provider):
            mapping = self.available_functions_provider()
        else:
            mapping = self.available_functions_provider
        return mapping if isinstance(mapping, dict) else {}

    async def _invoke_tool_by_name(self, tool_name: str, arguments: dict):
        available_functions = self._get_available_functions()
        target = available_functions.get(tool_name)
        if not target:
            return {"status": "failed", "error": f"Tool not found: {tool_name}"}

        try:
            if inspect.iscoroutinefunction(target):
                raw_result = await target(arguments)
            else:
                try:
                    raw_result = target(**arguments)
                except TypeError:
                    raw_result = target(arguments)
            result_dict = self.coerce_tool_result_to_dict(raw_result)
            if "status" not in result_dict:
                result_dict["status"] = "ok"
            return self._sanitize_payload(result_dict, max_chars=50000)
        except Exception as e:
            return {"status": "failed", "error": _redact_sensitive_text(str(e), max_chars=4000)}

    async def reasoning_plan(self, kwargs_dict):
        kwargs = kwargs_dict or {}
        goal = str(kwargs.get("goal", "")).strip()
        context = str(kwargs.get("context", "")).strip()
        max_steps = int(kwargs.get("max_steps", 8))
        max_steps = max(3, min(max_steps, 20))

        if not goal:
            return self._json_response({"status": "failed", "error": "goal is required"})

        separators = r"(?:\.\s+|;\s+|\n+| then | and then | after that )"
        pieces = [chunk.strip(" -\t\r\n") for chunk in re.split(separators, goal) if chunk.strip(" -\t\r\n")]
        if not pieces:
            pieces = [goal]

        steps = []
        for idx, piece in enumerate(pieces[:max_steps], start=1):
            steps.append({"step": idx, "action": piece})

        while len(steps) < min(3, max_steps):
            steps.append({"step": len(steps) + 1, "action": "Verify results and adjust based on observed output."})

        assumptions = []
        if context:
            assumptions.append("Provided context and constraints are accurate.")
        assumptions.extend(
            [
                "Required tools and permissions are available.",
                "Target paths and URLs are reachable.",
            ]
        )

        risks = [
            "Missing permissions or blocked resources can interrupt execution.",
            "Dynamic websites may require selector fallback and retries.",
        ]

        return self._json_response(
            {
                "status": "ok",
                "goal": goal,
                "context": context,
                "assumptions": assumptions,
                "plan_steps": steps[:max_steps],
                "risks": risks,
            }
        )

    async def workflow_execute(self, kwargs_dict):
        kwargs = kwargs_dict or {}
        steps = kwargs.get("steps", [])
        stop_on_error = bool(kwargs.get("stop_on_error", True))
        include_artifact = bool(kwargs.get("include_artifact", True))
        max_steps = int(kwargs.get("max_steps", 30))
        max_steps = max(1, min(max_steps, 100))

        if not isinstance(steps, list) or not steps:
            return self._json_response({"status": "failed", "error": "steps must be a non-empty array"})

        execution = []
        workflow_started = time.perf_counter()
        overall_status = "ok"

        for idx, step in enumerate(steps[:max_steps], start=1):
            if not isinstance(step, dict):
                execution.append({"step": idx, "status": "failed", "error": "Step must be object"})
                overall_status = "failed"
                if stop_on_error:
                    break
                continue

            tool_name = str(step.get("tool_name", "")).strip()
            arguments = step.get("arguments", {}) or {}
            required = bool(step.get("required", True))

            if not isinstance(arguments, dict):
                result = {"status": "failed", "error": "arguments must be an object"}
            elif not tool_name:
                result = {"status": "failed", "error": "tool_name is required"}
            elif tool_name in {"workflow_execute", "task_autopilot", "call_tool"}:
                result = {"status": "failed", "error": f"Orchestration tool '{tool_name}' is blocked inside workflow_execute to prevent recursion"}
            else:
                step_start = time.perf_counter()
                result = await self._invoke_tool_by_name(tool_name, arguments)
                result["duration_ms"] = int((time.perf_counter() - step_start) * 1000)

            status = result.get("status", "ok")
            entry = {
                "step": idx,
                "tool_name": tool_name,
                "required": required,
                "status": status,
                "result": result,
            }
            execution.append(entry)

            failed = status not in {"ok", "success"}
            if failed and required:
                overall_status = "failed"
                if stop_on_error:
                    break

        duration_ms = int((time.perf_counter() - workflow_started) * 1000)
        artifact = self._build_workflow_artifact(execution, overall_status, duration_ms)

        payload = {
            "status": overall_status,
            "executed_steps": len(execution),
            "duration_ms": duration_ms,
            "steps": execution,
            "developer_summary": artifact["developer_summary"],
        }
        if include_artifact:
            payload["artifact"] = artifact

        return self._json_response(payload)

    async def task_autopilot(self, kwargs_dict):
        kwargs = kwargs_dict or {}
        objective = str(kwargs.get("objective", "")).strip()
        path_value = kwargs.get("path", ".")
        max_focus_files = int(kwargs.get("max_focus_files", 6))
        include_preview = bool(kwargs.get("include_preview", False))
        include_artifact = bool(kwargs.get("include_artifact", True))
        max_focus_files = max(1, min(max_focus_files, 20))

        if not objective:
            return self._json_response({"status": "failed", "error": "objective is required"})

        plan = self.coerce_tool_result_to_dict(await self.reasoning_plan({"goal": objective, "context": f"path={path_value}"}))
        base = self.coerce_tool_result_to_dict(await self.codebase_analyze_fn({"path": path_value, "max_files": 2000, "top_n_large_files": 20}))
        if base.get("status") != "ok":
            return self._json_response(
                {
                    "status": "failed",
                    "objective": objective,
                    "plan": plan,
                    "analysis_error": base,
                }
            )

        focus_files = []
        key_files = base.get("key_files", [])
        large_files = [item.get("path") for item in base.get("largest_files", []) if isinstance(item, dict)]
        for candidate in key_files + large_files:
            if not candidate:
                continue
            candidate_path = Path(candidate)
            if not self.is_probably_text_source(candidate_path):
                continue
            if candidate not in focus_files:
                focus_files.append(candidate)
            if len(focus_files) >= max_focus_files:
                break

        deep_file_analysis = []
        for file_path in focus_files:
            analyzed = self.coerce_tool_result_to_dict(
                await self.fs_analyze_file_fn(
                    {
                        "path": file_path,
                        "max_chars": 250000,
                        "include_preview": include_preview,
                    }
                )
            )
            deep_file_analysis.append({"path": file_path, "analysis": analyzed})

        artifact = self._build_task_artifact(objective, path_value, plan, focus_files, deep_file_analysis, status="ok")

        payload = {
            "status": "ok",
            "objective": objective,
            "path": path_value,
            "plan": plan,
            "codebase": base,
            "focus_files": focus_files,
            "deep_file_analysis": deep_file_analysis,
            "developer_summary": artifact["developer_summary"],
            "next_actions": [
                "Refine target files and run fs_patch/fs_edit_lines for code modifications.",
                "Use workflow_execute to run deterministic multi-step tool sequences.",
                "Run compile/tests with run_command after modifications.",
            ],
        }
        if include_artifact:
            payload["artifact"] = artifact
        return self._json_response(payload)
