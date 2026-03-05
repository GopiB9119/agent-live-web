import json
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mcp_tools import MCPManager
from tooling.registry import register_or_update_tool_schema


RUN_MCP_LIVE_TESTS = os.getenv("RUN_MCP_LIVE_TESTS", "0").strip() == "1"


class MCPLiveIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        if not RUN_MCP_LIVE_TESTS:
            self.skipTest("Set RUN_MCP_LIVE_TESTS=1 to run live MCP integration tests.")

        self.manager = MCPManager(
            workspace_root=Path(__file__).resolve().parents[4],
            retryable_tools={
                "browser_navigate",
                "browser_click",
                "browser_type",
                "browser_fill_form",
                "browser_select_option",
                "browser_press_key",
                "browser_wait_for",
            },
            state_change_tools={
                "browser_click",
                "browser_type",
                "browser_fill_form",
                "browser_select_option",
                "browser_press_key",
            },
            ownership_skip_tools={"browser_tabs", "browser_close", "browser_install"},
        )
        self.agent_tools = []
        self.available_functions = {}

        await self.manager.init_mcp_client(
            agent_tools=self.agent_tools,
            available_functions=self.available_functions,
            register_or_update_tool_schema_fn=register_or_update_tool_schema,
        )
        if self.manager.mcp_session is None:
            self.fail("Live MCP session failed to initialize.")

    async def asyncTearDown(self):
        if hasattr(self, "manager") and self.manager:
            await self.manager.shutdown_mcp_client()

    async def test_session_reconnect_cycle(self):
        self.assertIsNotNone(self.manager.mcp_session)
        await self.manager.shutdown_mcp_client()
        self.assertIsNone(self.manager.mcp_session)

        await self.manager.init_mcp_client(
            agent_tools=self.agent_tools,
            available_functions=self.available_functions,
            register_or_update_tool_schema_fn=register_or_update_tool_schema,
        )
        self.assertIsNotNone(self.manager.mcp_session)

    async def test_tab_ownership_and_navigate_wrapper(self):
        navigate = self.available_functions.get("browser_navigate")
        self.assertTrue(callable(navigate), "browser_navigate wrapper is missing")

        payload = json.loads(await navigate({"url": "https://example.com"}))
        self.assertEqual(payload.get("tool"), "browser_navigate")
        self.assertIn(payload.get("status"), {"ok", "failed"})
        self.assertIn("attempts", payload)
        self.assertIn(int(payload.get("attempts", 0)), {1, 2})
        self.assertIn("verification", payload)

    async def test_retry_flow_for_retryable_click(self):
        click = self.available_functions.get("browser_click")
        self.assertTrue(callable(click), "browser_click wrapper is missing")

        payload = json.loads(await click({"element": "text:__definitely_not_present__"}))
        self.assertEqual(payload.get("tool"), "browser_click")
        self.assertEqual(int(payload.get("attempts", 0)), 2)
        self.assertIn(payload.get("status"), {"ok", "failed"})
        self.assertIn("verification", payload)


if __name__ == "__main__":
    unittest.main()
