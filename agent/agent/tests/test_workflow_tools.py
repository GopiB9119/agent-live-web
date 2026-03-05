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


def _is_text_source(_path):
    return True


class WorkflowManagerTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.mapping = {
            "echo": _async_echo,
            "add": _sync_add,
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


if __name__ == "__main__":
    unittest.main()
