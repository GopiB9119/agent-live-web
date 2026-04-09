import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mcp_tools import MCPManager


class _FakeItem:
    def __init__(self, text=None):
        self.text = text


class _FakeResult:
    def __init__(self, content=None, structured=None):
        self.content = content or []
        self.structuredContent = structured


class MCPManagerTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.manager = MCPManager(
            workspace_root=Path(self.tmp.name),
            retryable_tools={"browser_click"},
            state_change_tools={"browser_click"},
            ownership_skip_tools={"browser_tabs", "browser_close", "browser_install"},
        )

    def tearDown(self):
        self.tmp.cleanup()

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

    async def test_capture_debug_artifacts_collects_snapshot_and_output_listing(self):
        output_dir = Path(self.tmp.name) / "mcp-output"
        output_dir.mkdir(parents=True, exist_ok=True)
        trace_file = output_dir / "trace.zip"
        trace_file.write_bytes(b"trace-bytes")

        async def fake_list_tabs_state():
            return {
                "ok": True,
                "error": None,
                "tabs": [{"index": 0, "current": True, "title": "Example", "url": "https://example.com?token=abc123"}],
                "current": {"index": 0, "current": True, "title": "Example", "url": "https://example.com?token=abc123"},
                "raw": "- 0: (current) [Example](https://example.com?token=abc123)",
            }

        async def fake_call(tool_name, args=None):
            self.assertEqual(tool_name, "browser_snapshot")
            return {"ok": True, "text": "# Page snapshot\nAuthorization=secret-token", "error": None}

        self.manager._list_tabs_state = fake_list_tabs_state
        self.manager._call_mcp_tool_raw = fake_call
        self.manager.mcp_session = object()

        with patch.dict(os.environ, {"PLAYWRIGHT_MCP_OUTPUT_DIR": str(output_dir)}, clear=False):
            artifacts = await self.manager.capture_debug_artifacts(include_snapshot=True)

        self.assertEqual(artifacts["status"], "ok")
        self.assertTrue(artifacts["connected"])
        self.assertTrue(artifacts["tabs_state"]["ok"])
        self.assertTrue(artifacts["snapshot"]["ok"])
        self.assertIn("Page snapshot", artifacts["snapshot"]["text"])
        self.assertIn("Authorization=[REDACTED]", artifacts["snapshot"]["text"])
        self.assertNotIn("secret-token", artifacts["snapshot"]["text"])
        self.assertIn("token=[REDACTED]", artifacts["tabs_state"]["current"]["url"])
        self.assertEqual(artifacts["output_files"][0]["path"], "trace.zip")
        self.assertEqual(artifacts["output_files"][0]["size"], len(b"trace-bytes"))


if __name__ == "__main__":
    unittest.main()
