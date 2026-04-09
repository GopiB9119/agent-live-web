import json
import os
import re
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mcp_tools import MCPManager
from tooling.registry import register_or_update_tool_schema


RUN_MCP_LIVE_TESTS = os.getenv("RUN_MCP_LIVE_TESTS", "0").strip() == "1"
REPO_ROOT = Path(__file__).resolve().parents[4]
ARTIFACTS_ROOT = Path(os.getenv("MCP_LIVE_ARTIFACTS_DIR", str(REPO_ROOT / "logs" / "mcp-live"))).resolve()


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name, "1" if default else "0")
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


class MCPLiveIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        if not RUN_MCP_LIVE_TESTS:
            self.skipTest("Set RUN_MCP_LIVE_TESTS=1 to run live MCP integration tests.")

        self.manager = None
        self.agent_tools = []
        self.available_functions = {}
        self._env_restore = {}
        self.artifact_dir = self._build_artifact_dir()
        self._temp_output_root = tempfile.TemporaryDirectory(prefix="mcp-live-")
        self.output_dir = Path(self._temp_output_root.name) / "mcp-output"
        self._set_env_override("PLAYWRIGHT_MCP_OUTPUT_DIR", str(self.output_dir))
        if _env_flag("MCP_LIVE_CAPTURE_TRACE", True):
            self._set_env_override("PLAYWRIGHT_MCP_SAVE_TRACE", "true")

        self.manager = MCPManager(
            workspace_root=REPO_ROOT,
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

        await self.manager.init_mcp_client(
            agent_tools=self.agent_tools,
            available_functions=self.available_functions,
            register_or_update_tool_schema_fn=register_or_update_tool_schema,
        )
        if self.manager.mcp_session is None:
            self.fail("Live MCP session failed to initialize.")

    async def asyncTearDown(self):
        failed = self._current_test_failed()
        captured_artifacts = None
        try:
            if failed and self.manager:
                captured_artifacts = await self.manager.capture_debug_artifacts(include_snapshot=True)
        finally:
            if hasattr(self, "manager") and self.manager:
                await self.manager.shutdown_mcp_client()
            try:
                if failed:
                    self._write_failure_artifacts(captured_artifacts or {})
            finally:
                if hasattr(self, "_temp_output_root"):
                    self._temp_output_root.cleanup()
                self._restore_env_overrides()

    def _build_artifact_dir(self) -> Path:
        safe_test_name = re.sub(r"[^A-Za-z0-9_.-]+", "-", self.id())
        return ARTIFACTS_ROOT / safe_test_name

    def _set_env_override(self, name: str, value: str):
        if name not in self._env_restore:
            self._env_restore[name] = os.environ.get(name)
        os.environ[name] = value

    def _restore_env_overrides(self):
        for name, previous in self._env_restore.items():
            if previous is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = previous
        self._env_restore.clear()

    def _current_test_failed(self) -> bool:
        outcome = getattr(self, "_outcome", None)
        result = getattr(outcome, "result", None)
        if result is None:
            return False
        return any(case is self for case, _ in result.failures) or any(case is self for case, _ in result.errors)

    def _current_failure_details(self):
        outcome = getattr(self, "_outcome", None)
        result = getattr(outcome, "result", None)
        if result is None:
            return {"kind": "unknown", "traceback": ""}

        for case, err in result.failures:
            if case is self:
                return {"kind": "failure", "traceback": err}
        for case, err in result.errors:
            if case is self:
                return {"kind": "error", "traceback": err}
        return {"kind": "unknown", "traceback": ""}

    def _copy_output_files(self, source_dir: Path, destination_dir: Path):
        copied = []
        if not source_dir.exists():
            return copied

        for item in source_dir.rglob("*"):
            if not item.is_file():
                continue
            relative_path = item.relative_to(source_dir)
            target_path = destination_dir / relative_path
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, target_path)
            copied.append(relative_path.as_posix())
        return copied

    def _write_failure_artifacts(self, artifacts):
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        failure = self._current_failure_details()
        copy_raw_output_files = _env_flag("MCP_LIVE_COPY_RAW_OUTPUT_FILES", False)
        copied_output_files = (
            self._copy_output_files(self.output_dir, self.artifact_dir / "mcp-output")
            if copy_raw_output_files
            else []
        )
        metadata = {
            "test_id": self.id(),
            "artifact_dir": str(self.artifact_dir),
            "failure_kind": failure["kind"],
            "trace_capture_enabled": _env_flag("MCP_LIVE_CAPTURE_TRACE", True),
            "copy_raw_output_files": copy_raw_output_files,
            "mcp_output_dir": str(self.output_dir),
            "output_dir_exists": self.output_dir.exists(),
            "captured_output_files": artifacts.get("output_files", []),
            "copied_output_files": copied_output_files,
        }

        (self.artifact_dir / "failure-metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )
        (self.artifact_dir / "failure-traceback.txt").write_text(
            failure.get("traceback", ""),
            encoding="utf-8",
        )
        (self.artifact_dir / "tabs-state.json").write_text(
            json.dumps(artifacts.get("tabs_state", {}), indent=2, ensure_ascii=True),
            encoding="utf-8",
        )

        snapshot = artifacts.get("snapshot", {})
        snapshot_text = snapshot.get("text", "") if isinstance(snapshot, dict) else ""
        if snapshot_text:
            (self.artifact_dir / "browser-snapshot.md").write_text(snapshot_text, encoding="utf-8")

        print(f"[MCP Live Test] Failure artifacts saved to {self.artifact_dir}")

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
