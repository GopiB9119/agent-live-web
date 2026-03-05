import json
from pathlib import Path


class DiagnosticsManager:
    """
    Diagnostics and introspection manager for tool registry + health reports.
    """

    def __init__(
        self,
        agent_tools_provider,
        available_functions_provider,
        resolve_workspace_path_fn,
        to_bool_fn,
    ):
        self.agent_tools_provider = agent_tools_provider
        self.available_functions_provider = available_functions_provider
        self.resolve_workspace_path = resolve_workspace_path_fn
        self.to_bool = to_bool_fn

    def _get_agent_tools(self):
        tools = self.agent_tools_provider() if callable(self.agent_tools_provider) else self.agent_tools_provider
        return tools if isinstance(tools, list) else []

    def _get_available_functions(self):
        funcs = self.available_functions_provider() if callable(self.available_functions_provider) else self.available_functions_provider
        return funcs if isinstance(funcs, dict) else {}

    @staticmethod
    def _line_count(path_obj: Path) -> int:
        try:
            text = path_obj.read_text(encoding="utf-8", errors="replace")
            return 0 if text == "" else text.count("\n") + 1
        except Exception:
            return -1

    async def tool_catalog(self, kwargs_dict=None):
        kwargs = kwargs_dict or {}
        only_callable = bool(kwargs.get("only_callable", False))

        agent_tools = self._get_agent_tools()
        available_functions = self._get_available_functions()

        schema_map = {}
        for tool in agent_tools:
            fn = tool.get("function", {}) if isinstance(tool, dict) else {}
            name = fn.get("name")
            if name:
                schema_map[name] = {
                    "description": fn.get("description", ""),
                    "parameters": fn.get("parameters", {"type": "object", "properties": {}, "required": []}),
                }

        tools = []
        for name, meta in sorted(schema_map.items(), key=lambda item: item[0]):
            callable_now = name in available_functions
            if only_callable and not callable_now:
                continue
            tools.append(
                {
                    "name": name,
                    "callable": callable_now,
                    "description": meta["description"],
                    "parameters": meta["parameters"],
                }
            )

        return json.dumps(
            {
                "status": "ok",
                "count": len(tools),
                "tools": tools,
            },
            ensure_ascii=True,
        )

    async def agent_health_report(self, kwargs_dict=None):
        kwargs = kwargs_dict or {}
        include_tools = self.to_bool(kwargs.get("include_tools", False), False)
        fail_on_warn = self.to_bool(kwargs.get("fail_on_warn", False), False)
        line_budgets = kwargs.get("line_budgets", {}) if isinstance(kwargs.get("line_budgets", {}), dict) else {}
        if not line_budgets:
            line_budgets = {
                "agent/agent/agent.py": 900,
                "agent/agent/tools.py": 900,
                "agent/agent/SYSTEM_PROMPT.md": 250,
            }

        agent_tools = self._get_agent_tools()
        available_functions = self._get_available_functions()

        schema_names = []
        for tool in agent_tools:
            if not isinstance(tool, dict):
                continue
            fn = tool.get("function", {})
            name = fn.get("name")
            if isinstance(name, str) and name.strip():
                schema_names.append(name.strip())

        schema_counts = {}
        for name in schema_names:
            schema_counts[name] = schema_counts.get(name, 0) + 1
        duplicate_schema_names = sorted([name for name, count in schema_counts.items() if count > 1])

        callable_names = sorted([name for name in available_functions.keys() if isinstance(name, str) and name.strip()])
        schema_set = set(schema_names)
        callable_set = set(callable_names)

        schema_without_callable = sorted(schema_set - callable_set)
        callable_without_schema = sorted(callable_set - schema_set)

        line_report = []
        line_warnings = []
        for rel_path, budget in line_budgets.items():
            try:
                budget_int = int(budget)
            except Exception:
                budget_int = 0
            if budget_int <= 0:
                continue
            path_obj = self.resolve_workspace_path(str(rel_path), must_exist=False)
            count = self._line_count(path_obj) if path_obj.exists() else -1
            over_by = max(0, count - budget_int) if count >= 0 else 0
            row = {
                "path": str(rel_path),
                "exists": path_obj.exists(),
                "lines": count,
                "budget": budget_int,
                "over_by": over_by,
            }
            if count > budget_int:
                line_warnings.append(f"{rel_path} exceeds budget by {over_by} lines")
            line_report.append(row)

        issues = []
        warnings = []
        if duplicate_schema_names:
            issues.append(f"Duplicate tool schemas: {', '.join(duplicate_schema_names[:12])}")
        if schema_without_callable:
            issues.append(f"Tool schemas without callable mapping: {', '.join(schema_without_callable[:12])}")
        if line_warnings:
            warnings.extend(line_warnings[:20])
        if callable_without_schema:
            warnings.append(f"Callable functions without explicit schema: {len(callable_without_schema)}")

        status = "ok"
        if issues:
            status = "failed"
        elif warnings:
            status = "warn"
        if fail_on_warn and status == "warn":
            status = "failed"

        recommendations = [
            "Keep function schemas and AVAILABLE_FUNCTIONS in sync.",
            "Keep tools.py focused on wiring; place domain logic in dedicated modules.",
            "Run this check before release and after adding new tools.",
        ]
        if any(row.get("path") == "agent/agent/tools.py" and row.get("over_by", 0) > 0 for row in line_report):
            recommendations.insert(
                0,
                "Split tools.py into modules: fs_tools, memory_tools, web_tools, oauth_tools, mcp_tools, command_tools, workflow_tools, and diagnostics_tools.",
            )

        payload = {
            "status": status,
            "summary": {
                "tool_schemas": len(schema_names),
                "callable_functions": len(callable_names),
                "duplicate_schema_names": len(duplicate_schema_names),
                "schema_without_callable": len(schema_without_callable),
                "callable_without_schema": len(callable_without_schema),
                "line_warnings": len(line_warnings),
            },
            "issues": issues,
            "warnings": warnings,
            "line_report": line_report,
            "recommendations": recommendations,
        }
        if include_tools:
            payload["tool_names"] = {
                "schemas": sorted(schema_set),
                "callables": callable_names,
            }
        return json.dumps(payload, ensure_ascii=True)
