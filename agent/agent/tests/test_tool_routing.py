import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import tools as tools_module
from capability_router import activate_turn_tool_context, reset_turn_tool_context


class ToolRoutingTests(unittest.IsolatedAsyncioTestCase):
    async def test_execute_tool_with_policy_blocks_tool_not_exposed_for_turn(self):
        token = activate_turn_tool_context(
            {
                "allowed_tool_names": ["calculate"],
                "surfaces": ["core"],
                "task_spec": {"task_mode": "inspect"},
            }
        )
        try:
            with patch.object(tools_module, "write_safety_event") as write_event:
                raw = await tools_module.execute_tool_with_policy("fs_read", {"path": "README.md"})
        finally:
            reset_turn_tool_context(token)

        result = json.loads(raw)
        self.assertEqual(result["status"], "blocked")
        self.assertEqual(result["reason_code"], "tool_not_exposed_for_turn")
        self.assertEqual(result["tool"], "fs_read")
        write_event.assert_called_once()


if __name__ == "__main__":
    unittest.main()
