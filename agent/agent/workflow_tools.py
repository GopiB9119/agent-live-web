import inspect
import json
import re
import time
from pathlib import Path


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
    def coerce_tool_result_to_dict(raw_result):
        if isinstance(raw_result, dict):
            return raw_result
        text = str(raw_result)
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
        return {"status": "ok", "raw": text}

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
            return result_dict
        except Exception as e:
            return {"status": "failed", "error": str(e)}

    async def reasoning_plan(self, kwargs_dict):
        kwargs = kwargs_dict or {}
        goal = str(kwargs.get("goal", "")).strip()
        context = str(kwargs.get("context", "")).strip()
        max_steps = int(kwargs.get("max_steps", 8))
        max_steps = max(3, min(max_steps, 20))

        if not goal:
            return json.dumps({"status": "failed", "error": "goal is required"}, ensure_ascii=True)

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

        return json.dumps(
            {
                "status": "ok",
                "goal": goal,
                "context": context,
                "assumptions": assumptions,
                "plan_steps": steps[:max_steps],
                "risks": risks,
            },
            ensure_ascii=True,
        )

    async def workflow_execute(self, kwargs_dict):
        kwargs = kwargs_dict or {}
        steps = kwargs.get("steps", [])
        stop_on_error = bool(kwargs.get("stop_on_error", True))
        max_steps = int(kwargs.get("max_steps", 30))
        max_steps = max(1, min(max_steps, 100))

        if not isinstance(steps, list) or not steps:
            return json.dumps({"status": "failed", "error": "steps must be a non-empty array"}, ensure_ascii=True)

        execution = []
        workflow_started = time.time()
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
            elif tool_name in {"workflow_execute"}:
                result = {"status": "failed", "error": "Recursive workflow_execute is blocked"}
            else:
                step_start = time.time()
                result = await self._invoke_tool_by_name(tool_name, arguments)
                result["duration_ms"] = int((time.time() - step_start) * 1000)

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

        return json.dumps(
            {
                "status": overall_status,
                "executed_steps": len(execution),
                "duration_ms": int((time.time() - workflow_started) * 1000),
                "steps": execution,
            },
            ensure_ascii=True,
        )

    async def task_autopilot(self, kwargs_dict):
        kwargs = kwargs_dict or {}
        objective = str(kwargs.get("objective", "")).strip()
        path_value = kwargs.get("path", ".")
        max_focus_files = int(kwargs.get("max_focus_files", 6))
        include_preview = bool(kwargs.get("include_preview", False))
        max_focus_files = max(1, min(max_focus_files, 20))

        if not objective:
            return json.dumps({"status": "failed", "error": "objective is required"}, ensure_ascii=True)

        plan = self.coerce_tool_result_to_dict(await self.reasoning_plan({"goal": objective, "context": f"path={path_value}"}))
        base = self.coerce_tool_result_to_dict(await self.codebase_analyze_fn({"path": path_value, "max_files": 2000, "top_n_large_files": 20}))
        if base.get("status") != "ok":
            return json.dumps(
                {
                    "status": "failed",
                    "objective": objective,
                    "plan": plan,
                    "analysis_error": base,
                },
                ensure_ascii=True,
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

        return json.dumps(
            {
                "status": "ok",
                "objective": objective,
                "path": path_value,
                "plan": plan,
                "codebase": base,
                "focus_files": focus_files,
                "deep_file_analysis": deep_file_analysis,
                "next_actions": [
                    "Refine target files and run fs_patch/fs_edit_lines for code modifications.",
                    "Use workflow_execute to run deterministic multi-step tool sequences.",
                    "Run compile/tests with run_command after modifications.",
                ],
            },
            ensure_ascii=True,
        )
