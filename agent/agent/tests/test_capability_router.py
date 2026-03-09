import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from capability_router import (
    activate_turn_tool_context,
    build_turn_tool_context,
    get_turn_tool_context,
    reset_turn_tool_context,
)
from task_spec import build_task_spec


def _schema(name: str):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": f"Tool: {name}",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    }


class CapabilityRouterTests(unittest.TestCase):
    def setUp(self):
        self.agent_tools = [
            _schema("calculate"),
            _schema("reasoning_plan"),
            _schema("task_autopilot"),
            _schema("workflow_execute"),
            _schema("tool_catalog"),
            _schema("fs_read"),
            _schema("fs_write"),
            _schema("run_command"),
            _schema("browser_navigate"),
            _schema("browser_click"),
            _schema("browser_snapshot"),
            _schema("agent_proxy_status"),
            _schema("call_tool"),
        ]
        self.available_tool_names = [
            "calculate",
            "reasoning_plan",
            "task_autopilot",
            "workflow_execute",
            "tool_catalog",
            "fs_read",
            "fs_write",
            "run_command",
            "browser_navigate",
            "browser_click",
            "browser_snapshot",
            "agent_proxy_status",
            "call_tool",
        ]

    def test_router_limits_repo_inspect_task_to_workspace_and_core_tools(self):
        task_spec = build_task_spec("Read this repo and understand the failing code paths.")
        context = build_turn_tool_context(
            task_spec,
            self.agent_tools,
            self.available_tool_names,
            mcp_status={"connected": True},
        )
        self.assertIn("workspace", context["surfaces"])
        self.assertIn("diagnostics", context["surfaces"])
        self.assertIn("fs_read", context["allowed_tool_names"])
        self.assertNotIn("browser_click", context["allowed_tool_names"])
        self.assertNotIn("call_tool", context["allowed_tool_names"])

    def test_router_enables_browser_tools_for_browser_task(self):
        task_spec = build_task_spec("Use Playwright to navigate the website, click the form, and capture a screenshot.")
        context = build_turn_tool_context(
            task_spec,
            self.agent_tools,
            self.available_tool_names,
            mcp_status={"connected": True},
        )
        self.assertIn("browser", context["surfaces"])
        self.assertIn("browser_navigate", context["allowed_tool_names"])
        self.assertIn("browser_click", context["allowed_tool_names"])
        self.assertIn("browser_snapshot", context["allowed_tool_names"])

    def test_turn_tool_context_round_trips_through_contextvar(self):
        context = {"allowed_tool_names": ["calculate"], "surfaces": ["core"], "task_spec": {"task_mode": "inspect"}}
        token = activate_turn_tool_context(context)
        try:
            self.assertEqual(get_turn_tool_context()["allowed_tool_names"], ["calculate"])
        finally:
            reset_turn_tool_context(token)
        self.assertEqual(get_turn_tool_context(), {})


if __name__ == "__main__":
    unittest.main()
