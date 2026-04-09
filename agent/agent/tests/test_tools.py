import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import tools


class ToolsModuleTests(unittest.IsolatedAsyncioTestCase):
    async def test_call_tool_sanitizes_stringified_result(self):
        original_functions = dict(tools.AVAILABLE_FUNCTIONS)

        async def _secret_tool(_kwargs):
            return json.dumps(
                {
                    "status": "ok",
                    "access_token": "abc123",
                    "auth": {"mode": "oauth_profile"},
                    "url": "https://example.com/api?token=abc123",
                }
            )

        tools.AVAILABLE_FUNCTIONS["secret_tool"] = _secret_tool
        try:
            raw = await tools.call_tool({"tool_name": "secret_tool", "arguments": {}})
            result = json.loads(raw)
            self.assertEqual(result["status"], "ok")
            self.assertIn('"access_token": "[REDACTED]"', result["result"])
            self.assertIn('"mode": "oauth_profile"', result["result"])
            self.assertIn('token=[REDACTED]', result["result"])
        finally:
            tools.AVAILABLE_FUNCTIONS.clear()
            tools.AVAILABLE_FUNCTIONS.update(original_functions)

    async def test_call_tool_blocks_orchestration_recursion(self):
        """call_tool must refuse to invoke itself, workflow_execute, and task_autopilot."""
        for blocked_name in ("call_tool", "workflow_execute", "task_autopilot"):
            raw = await tools.call_tool({"tool_name": blocked_name, "arguments": {}})
            result = json.loads(raw)
            self.assertEqual(result["status"], "failed", f"{blocked_name} should be blocked")
            self.assertIn("recursion", result["error"].lower(), f"{blocked_name} error should mention recursion")

    async def test_call_tool_missing_tool_name_fails(self):
        raw = await tools.call_tool({"tool_name": "", "arguments": {}})
        result = json.loads(raw)
        self.assertEqual(result["status"], "failed")
        self.assertIn("required", result["error"].lower())

    async def test_call_tool_unknown_tool_fails(self):
        raw = await tools.call_tool({"tool_name": "nonexistent_tool_xyz", "arguments": {}})
        result = json.loads(raw)
        self.assertEqual(result["status"], "failed")
        self.assertIn("not found", result["error"].lower())

    async def test_call_tool_invalid_arguments_type_fails(self):
        raw = await tools.call_tool({"tool_name": "calculate", "arguments": "not a dict"})
        result = json.loads(raw)
        self.assertEqual(result["status"], "failed")
        self.assertIn("arguments must be an object", result["error"])

    async def test_call_tool_invokes_calculate_correctly(self):
        raw = await tools.call_tool({"tool_name": "calculate", "arguments": {"expression": "2+3"}})
        result = json.loads(raw)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["tool_name"], "calculate")
        self.assertIn("5", result["result"])

    async def test_call_tool_handles_tool_exception(self):
        original = dict(tools.AVAILABLE_FUNCTIONS)
        async def _crashing_tool(_kwargs):
            raise ValueError("intentional test error")
        tools.AVAILABLE_FUNCTIONS["crashing_tool"] = _crashing_tool
        try:
            raw = await tools.call_tool({"tool_name": "crashing_tool", "arguments": {}})
            result = json.loads(raw)
            self.assertEqual(result["status"], "failed")
            self.assertIn("intentional test error", result["error"])
        finally:
            tools.AVAILABLE_FUNCTIONS.clear()
            tools.AVAILABLE_FUNCTIONS.update(original)

    def test_calculate_basic_arithmetic(self):
        self.assertEqual(tools.calculate("2+3"), "5")
        self.assertEqual(tools.calculate("10*5"), "50")
        self.assertEqual(tools.calculate("(3+4)*2"), "14")
        self.assertEqual(tools.calculate("-5+3"), "-2")

    def test_calculate_rejects_code_injection(self):
        result = tools.calculate("__import__('os').system('ls')")
        self.assertTrue(result.startswith("Error"))

    def test_calculate_rejects_builtins(self):
        result = tools.calculate("open('/etc/passwd')")
        self.assertTrue(result.startswith("Error"))

    def test_calculate_handles_division_by_zero(self):
        result = tools.calculate("1/0")
        self.assertTrue(result.startswith("Error"))
