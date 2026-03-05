import json
import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from diagnostics_tools import DiagnosticsManager


class DiagnosticsManagerTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "agent" / "agent").mkdir(parents=True, exist_ok=True)
        (self.root / "agent" / "agent" / "agent.py").write_text("print('ok')\n", encoding="utf-8")
        (self.root / "agent" / "agent" / "tools.py").write_text("print('ok')\n", encoding="utf-8")
        (self.root / "agent" / "agent" / "SYSTEM_PROMPT.md").write_text("# prompt\n", encoding="utf-8")

        self.agent_tools = [
            {
                "type": "function",
                "function": {
                    "name": "alpha",
                    "description": "alpha tool",
                    "parameters": {"type": "object", "properties": {}, "required": []},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "beta",
                    "description": "beta tool",
                    "parameters": {"type": "object", "properties": {}, "required": []},
                },
            },
        ]
        self.callables = {"alpha": object(), "gamma": object()}

        def _resolver(raw_path, must_exist=False):
            path = (self.root / raw_path).resolve()
            if must_exist and not path.exists():
                raise FileNotFoundError(str(path))
            return path

        self.manager = DiagnosticsManager(
            agent_tools_provider=lambda: self.agent_tools,
            available_functions_provider=lambda: self.callables,
            resolve_workspace_path_fn=_resolver,
            to_bool_fn=lambda value, default=False: default if value is None else bool(value),
        )

    def tearDown(self):
        self.tmp.cleanup()

    async def test_tool_catalog_only_callable(self):
        raw = await self.manager.tool_catalog({"only_callable": True})
        result = json.loads(raw)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["count"], 1)
        self.assertEqual(result["tools"][0]["name"], "alpha")

    async def test_agent_health_report_warn_on_drift(self):
        raw = await self.manager.agent_health_report({"include_tools": True})
        result = json.loads(raw)
        self.assertEqual(result["status"], "failed")
        self.assertTrue(result["summary"]["schema_without_callable"] >= 1)
        self.assertTrue(result["summary"]["callable_without_schema"] >= 1)
        self.assertIn("tool_names", result)

    async def test_agent_health_report_fail_on_warn(self):
        # Align schema/callable sets, then force warn via line budget.
        self.callables = {"alpha": object(), "beta": object()}
        raw = await self.manager.agent_health_report(
            {
                "fail_on_warn": True,
                "line_budgets": {"agent/agent/tools.py": 1},
            }
        )
        result = json.loads(raw)
        self.assertEqual(result["status"], "failed")


if __name__ == "__main__":
    unittest.main()
