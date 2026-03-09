import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mcp_tools import MCPManager
from tests_support import create_repo_local_temp_dir, remove_tree


class _FakeItem:
    def __init__(self, text=None):
        self.text = text


class _FakeResult:
    def __init__(self, content=None, structured=None):
        self.content = content or []
        self.structuredContent = structured


class MCPManagerTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp_path = create_repo_local_temp_dir(Path(__file__), "test-mcp-tools", "mcp-tools")
        self.manager = MCPManager(
            workspace_root=self.tmp_path,
            retryable_tools={"browser_click"},
            state_change_tools={"browser_click"},
            ownership_skip_tools={"browser_tabs", "browser_close", "browser_install"},
        )

    def tearDown(self):
        remove_tree(self.tmp_path)

    async def test_parse_tabs_text(self):
        text = "- 0: (current) [Home](https://example.com)\n- 1: [Blank](about:blank)"
        tabs = self.manager._parse_tabs_text(text)
        self.assertEqual(len(tabs), 2)
        self.assertTrue(tabs[0]["current"])
        self.assertEqual(tabs[1]["url"], "about:blank")

    async def test_hosts_match(self):
        self.assertTrue(self.manager._hosts_match("https://example.com/x", "https://sub.example.com/y"))
        self.assertTrue(self.manager._hosts_match("https://example.com/x", "https://example.com/y"))
        self.assertFalse(self.manager._hosts_match("https://example.com/x", "https://another.com/y"))

    async def test_serialize_call_result(self):
        result = _FakeResult(content=[_FakeItem("alpha"), _FakeItem("beta")])
        text = self.manager._serialize_call_result(result)
        self.assertIn("alpha", text)
        self.assertIn("beta", text)

        structured_only = _FakeResult(content=[], structured={"status": "ok"})
        structured_text = self.manager._serialize_call_result(structured_only)
        self.assertIn('"status": "ok"', structured_text)

    async def test_format_step_response(self):
        response = json.loads(
            self.manager._format_step_response(
                "browser_click",
                {"element": "text:Submit"},
                2,
                {"ok": True, "reason": "verified"},
                {"ok": True, "text": "done", "error": None},
                recovered=True,
            )
        )
        self.assertEqual(response["status"], "ok")
        self.assertEqual(response["attempts"], 2)
        self.assertTrue(response["recovered"])

    async def test_browser_methods_fail_when_disconnected(self):
        tabs = json.loads(await self.manager.browser_tabs_list())
        self.assertEqual(tabs["status"], "failed")
        selected = json.loads(await self.manager.browser_tab_select({"index": 0}))
        self.assertEqual(selected["status"], "failed")

    async def test_timeout_configuration_reads_environment(self):
        previous = os.environ.get("AGENT_MCP_TOOL_TIMEOUT_SEC")
        os.environ["AGENT_MCP_TOOL_TIMEOUT_SEC"] = "12.5"
        try:
            configured = MCPManager(workspace_root=self.tmp_path)
        finally:
            if previous is None:
                os.environ.pop("AGENT_MCP_TOOL_TIMEOUT_SEC", None)
            else:
                os.environ["AGENT_MCP_TOOL_TIMEOUT_SEC"] = previous
        self.assertEqual(configured.tool_timeout_seconds, 12.5)

    async def test_runtime_status_reports_last_connection_state(self):
        status = self.manager.runtime_status()
        self.assertFalse(status["connected"])
        self.assertEqual(status["status"], "not_attempted")
        self.assertEqual(status["tool_count"], 0)
        self.assertEqual(status["proxy_status"], {})

        self.manager.last_connect_status = "failed"
        self.manager.last_connect_error = "Access is denied"
        failed = self.manager.runtime_status()
        self.assertEqual(failed["status"], "failed")
        self.assertEqual(failed["error"], "Access is denied")

    async def test_refresh_proxy_status_caches_direct_mcp_summary(self):
        self.manager.mcp_session = object()
        self.manager._call_mcp_tool_raw = AsyncMock(
            return_value={
                "ok": True,
                "structured": {
                    "runtime_status": "ready",
                    "startup_trust": "direct MCP proxy",
                    "resume_state": "not applicable in direct MCP mode",
                    "summary": "Guarded direct MCP mode is ready.",
                },
            }
        )

        payload = await self.manager._refresh_proxy_status()

        self.assertEqual(payload["runtime_status"], "ready")
        status = self.manager.runtime_status()
        self.assertEqual(status["proxy_status"]["startup_trust"], "direct MCP proxy")

    async def test_verify_step_accepts_agent_proxy_status_without_browser_context(self):
        verification = await self.manager._verify_step(
            "agent_proxy_status",
            {},
            {},
            {
                "ok": True,
                "structured": {
                    "runtime_status": "ready",
                    "summary": "Guarded direct MCP mode is ready.",
                },
            },
        )
        self.assertTrue(verification["ok"])
        self.assertEqual(verification["details"]["proxy_status"]["runtime_status"], "ready")


if __name__ == "__main__":
    unittest.main()
