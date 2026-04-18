import json
import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from workflow_tools import WorkflowManager


async def _async_echo(kwargs):
    return json.dumps({"status": "ok", "echo": kwargs})


def _sync_add(a=0, b=0):
    return json.dumps({"status": "ok", "sum": int(a) + int(b)})


async def _codebase_ok(_kwargs):
    return {
        "status": "ok",
        "key_files": ["README.md", "agent/agent/tools.py"],
        "largest_files": [{"path": "agent/agent/agent.py"}],
    }


async def _analyze_ok(kwargs):
    return {"status": "ok", "path": kwargs.get("path"), "language": "python"}


async def _secret_tool(_kwargs):
    return json.dumps(
        {
            "status": "ok",
            "access_token": "abc123",
            "auth": {"mode": "oauth_profile", "token_preview": "abc***"},
            "url": "https://example.com/api?token=abc123",
        }
    )


def _is_text_source(_path):
    return True


class WorkflowManagerTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.mapping = {
            "echo": _async_echo,
            "add": _sync_add,
            "secret": _secret_tool,
        }
        self.manager = WorkflowManager(
            available_functions_provider=lambda: self.mapping,
            is_probably_text_source_fn=_is_text_source,
            codebase_analyze_fn=_codebase_ok,
            fs_analyze_file_fn=_analyze_ok,
        )

    async def test_reasoning_plan_requires_goal(self):
        raw = await self.manager.reasoning_plan({})
        result = json.loads(raw)
        self.assertEqual(result["status"], "failed")

    async def test_workflow_execute_success(self):
        raw = await self.manager.workflow_execute(
            {
                "steps": [
                    {"tool_name": "echo", "arguments": {"x": 1}},
                    {"tool_name": "add", "arguments": {"a": 2, "b": 3}},
                ]
            }
        )
        result = json.loads(raw)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["executed_steps"], 2)
        self.assertIn("developer_summary", result)
        self.assertEqual(result["artifact"]["kind"], "workflow_execution")
        self.assertEqual(len(result["artifact"]["step_summaries"]), 2)

    async def test_workflow_execute_required_failure_stops(self):
        raw = await self.manager.workflow_execute(
            {
                "steps": [
                    {"tool_name": "missing_tool", "arguments": {}, "required": True},
                    {"tool_name": "echo", "arguments": {"x": 1}},
                ],
                "stop_on_error": True,
            }
        )
        result = json.loads(raw)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["executed_steps"], 1)

    async def test_task_autopilot_happy_path(self):
        raw = await self.manager.task_autopilot({"objective": "analyze project", "path": "."})
        result = json.loads(raw)
        self.assertEqual(result["status"], "ok")
        self.assertTrue(len(result["focus_files"]) >= 1)
        self.assertTrue(len(result["deep_file_analysis"]) >= 1)
        self.assertEqual(result["artifact"]["kind"], "task_autopilot")
        self.assertIn("developer_summary", result)

    async def test_workflow_execute_sanitizes_nested_tool_results(self):
        raw = await self.manager.workflow_execute(
            {
                "steps": [
                    {"tool_name": "secret", "arguments": {}},
                ]
            }
        )
        result = json.loads(raw)
        step_result = result["steps"][0]["result"]
        self.assertEqual(result["status"], "ok")
        self.assertEqual(step_result["access_token"], "[REDACTED]")
        self.assertEqual(step_result["auth"]["mode"], "oauth_profile")
        self.assertIn("token=[REDACTED]", step_result["url"])

    async def test_reasoning_plan_sanitizes_goal_text(self):
        raw = await self.manager.reasoning_plan({"goal": "check https://example.com?token=abc123 and report"})
        result = json.loads(raw)
        self.assertEqual(result["status"], "ok")
        self.assertIn("token=[REDACTED]", result["goal"])

    async def test_workflow_execute_can_disable_artifact(self):
        raw = await self.manager.workflow_execute(
            {
                "steps": [{"tool_name": "echo", "arguments": {"x": 1}}],
                "include_artifact": False,
            }
        )
        result = json.loads(raw)
        self.assertEqual(result["status"], "ok")
        self.assertNotIn("artifact", result)

    async def test_workflow_execute_blocks_orchestration_recursion(self):
        """workflow_execute must block itself, task_autopilot, and call_tool."""
        for blocked in ("workflow_execute", "task_autopilot", "call_tool"):
            raw = await self.manager.workflow_execute(
                {"steps": [{"tool_name": blocked, "arguments": {}, "required": True}], "stop_on_error": True}
            )
            result = json.loads(raw)
            self.assertEqual(result["status"], "failed", f"{blocked} should be blocked")
            step_result = result["steps"][0]["result"]
            self.assertIn("recursion", step_result["error"].lower(), f"{blocked} error should mention recursion")


if __name__ == "__main__":
    unittest.main()
