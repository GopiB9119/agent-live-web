import json
import os
import sys
import unittest
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, cast

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tests_support import create_temp_dir_under, remove_tree

RUN_MCP_LIVE_TESTS = os.getenv("RUN_MCP_LIVE_TESTS", "0").strip() == "1"
ToolCallable = Callable[[Dict[str, Any]], Awaitable[str]]
RegisterSchemaFn = Callable[[List[Dict[str, Any]], str, str, Dict[str, Any]], None]


def _read_timeout_seconds(env_name: str, default: float) -> float:
    raw_value = str(os.environ.get(env_name, str(default))).strip()
    try:
        parsed = float(raw_value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


LIVE_INIT_TIMEOUT_SEC = _read_timeout_seconds("PLAYWRIGHT_MCP_LIVE_INIT_TIMEOUT_SEC", 60.0)
LIVE_CALL_TIMEOUT_SEC = _read_timeout_seconds("PLAYWRIGHT_MCP_LIVE_CALL_TIMEOUT_SEC", 45.0)
LIVE_KEEP_RUNTIME = os.getenv("PLAYWRIGHT_MCP_LIVE_KEEP_RUNTIME", "0").strip().lower() in {"1", "true", "yes", "on"}


class _BaseMCPLiveIntegrationTest(unittest.IsolatedAsyncioTestCase):
    workspace_root: Path
    runtime_root: Optional[Path]
    output_dir: Path

    def setUp(self):
        self.workspace_root = Path(__file__).resolve().parents[4]
        self.runtime_root = None
        self.output_dir = self.workspace_root / ".playwright-mcp" / "output"
        self._env_backup: Dict[str, Optional[str]] = {}
        if RUN_MCP_LIVE_TESTS:
            self._configure_live_runtime(self._runtime_prefix())

    def _runtime_prefix(self) -> str:
        return "mcp-live"

    def _set_env_override(self, key: str, value: str):
        if key not in self._env_backup:
            self._env_backup[key] = os.environ.get(key)
        os.environ[key] = value

    def _configure_live_runtime(self, prefix: str):
        base_dir = Path(
            os.environ.get(
                "PLAYWRIGHT_MCP_LIVE_RUNTIME_ROOT",
                str(self.workspace_root / ".playwright-mcp" / "live-tests"),
            )
        ).resolve()
        self.runtime_root = create_temp_dir_under(base_dir, prefix)

        owner_name = str(os.environ.get("PLAYWRIGHT_MCP_OWNER", f"{prefix}-{os.getpid()}")).strip() or f"{prefix}-{os.getpid()}"
        owner_file = Path(os.environ.get("PLAYWRIGHT_MCP_OWNER_FILE", str(self.runtime_root / "active-owner.txt")))
        user_data_dir = Path(os.environ.get("PLAYWRIGHT_MCP_USER_DATA_DIR", str(self.runtime_root / "edge-profile")))
        self.output_dir = Path(os.environ.get("PLAYWRIGHT_MCP_OUTPUT_DIR", str(self.runtime_root / "output")))

        self._set_env_override("PLAYWRIGHT_MCP_OWNER", owner_name)
        self._set_env_override("PLAYWRIGHT_MCP_FORCE_OWNER", os.environ.get("PLAYWRIGHT_MCP_FORCE_OWNER", "true"))
        self._set_env_override("PLAYWRIGHT_MCP_HEADLESS", os.environ.get("PLAYWRIGHT_MCP_HEADLESS", "true"))
        self._set_env_override("PLAYWRIGHT_MCP_OWNER_FILE", str(owner_file))
        self._set_env_override("PLAYWRIGHT_MCP_USER_DATA_DIR", str(user_data_dir))
        self._set_env_override("PLAYWRIGHT_MCP_OUTPUT_DIR", str(self.output_dir))
        self._set_env_override(
            "PLAYWRIGHT_MCP_PROXY_INIT_TIMEOUT_MS",
            os.environ.get("PLAYWRIGHT_MCP_PROXY_INIT_TIMEOUT_MS", str(int(LIVE_INIT_TIMEOUT_SEC * 1000))),
        )
        self._set_env_override(
            "PLAYWRIGHT_MCP_PROXY_REQUEST_TIMEOUT_MS",
            os.environ.get("PLAYWRIGHT_MCP_PROXY_REQUEST_TIMEOUT_MS", str(int(LIVE_CALL_TIMEOUT_SEC * 1000))),
        )
        self._set_env_override(
            "AGENT_MCP_CONNECT_TIMEOUT_SEC",
            os.environ.get("AGENT_MCP_CONNECT_TIMEOUT_SEC", str(LIVE_INIT_TIMEOUT_SEC)),
        )
        self._set_env_override(
            "AGENT_MCP_TOOL_TIMEOUT_SEC",
            os.environ.get("AGENT_MCP_TOOL_TIMEOUT_SEC", str(LIVE_CALL_TIMEOUT_SEC)),
        )

    def _restore_live_runtime(self):
        for key, original_value in reversed(list(self._env_backup.items())):
            if original_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = original_value
        self._env_backup.clear()

        if self.runtime_root is not None and not LIVE_KEEP_RUNTIME:
            remove_tree(self.runtime_root)
        self.runtime_root = None


class MCPLiveIntegrationTests(_BaseMCPLiveIntegrationTest):
    manager: Optional[Any]
    agent_tools: List[Dict[str, Any]]
    available_functions: Dict[str, ToolCallable]
    register_or_update_tool_schema_fn: Optional[RegisterSchemaFn]

    def setUp(self):
        super().setUp()
        self.manager = None
        self.agent_tools = []
        self.available_functions = {}
        self.register_or_update_tool_schema_fn = None

    def _runtime_prefix(self) -> str:
        return "mcp-manager-live"

    def _manager(self) -> Any:
        self.assertIsNotNone(self.manager, "MCP manager was not initialized")
        return self.manager

    def _register_schema_fn(self) -> RegisterSchemaFn:
        self.assertIsNotNone(self.register_or_update_tool_schema_fn, "register_or_update_tool_schema was not initialized")
        return cast(RegisterSchemaFn, self.register_or_update_tool_schema_fn)

    async def asyncSetUp(self):
        if not RUN_MCP_LIVE_TESTS:
            self.skipTest("Set RUN_MCP_LIVE_TESTS=1 to run live MCP integration tests.")

        from mcp_tools import MCPManager  # pyright: ignore[reportMissingImports]
        from tooling.registry import register_or_update_tool_schema  # pyright: ignore[reportMissingImports]

        self.register_or_update_tool_schema_fn = register_or_update_tool_schema
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

        try:
            await self._manager().init_mcp_client(
                agent_tools=self.agent_tools,
                available_functions=self.available_functions,
                register_or_update_tool_schema_fn=self._register_schema_fn(),
            )
            if self._manager().mcp_session is None:
                self.fail("Live MCP session failed to initialize.")
        except Exception:
            self._restore_live_runtime()
            raise

    async def asyncTearDown(self):
        try:
            manager = self.manager
            if manager is not None:
                await manager.shutdown_mcp_client()
            self.manager = None
        finally:
            self._restore_live_runtime()

    async def _call_tool_json(self, tool_name: str, args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        tool_fn = self.available_functions.get(tool_name)
        self.assertIsNotNone(tool_fn, f"{tool_name} wrapper is missing")
        response = await cast(ToolCallable, tool_fn)(args or {})
        payload = json.loads(response)
        self.assertIsInstance(payload, dict, f"{tool_name} returned non-object JSON payload")
        return cast(Dict[str, Any], payload)

    async def test_session_reconnect_cycle(self):
        manager = self._manager()
        self.assertIsNotNone(manager.mcp_session)
        for _ in range(2):
            await manager.shutdown_mcp_client()
            self.assertIsNone(manager.mcp_session)

            await manager.init_mcp_client(
                agent_tools=self.agent_tools,
                available_functions=self.available_functions,
                register_or_update_tool_schema_fn=self._register_schema_fn(),
            )
            self.assertIsNotNone(manager.mcp_session)
            self.assertTrue(callable(self.available_functions.get("browser_navigate")))
            self.assertTrue(callable(self.available_functions.get("browser_tabs")))
            self.assertTrue(callable(self.available_functions.get("agent_proxy_status")))

    async def test_manager_proxy_status_tool_reports_readiness_and_trust(self):
        payload = await self._call_tool_json("agent_proxy_status", {})
        self.assertEqual(payload.get("tool"), "agent_proxy_status")
        self.assertEqual(payload.get("status"), "ok")
        self.assertEqual(payload.get("args"), {})
        self.assertEqual(payload.get("attempts"), 1)
        self.assertFalse(bool(payload.get("recovered", False)))
        verification = payload.get("verification", {})
        self.assertIsInstance(verification, dict)
        self.assertTrue(bool(verification.get("ok", False)))
        details = verification.get("details", {})
        self.assertIsInstance(details, dict)
        proxy_status = details.get("proxy_status", {})
        self.assertIsInstance(proxy_status, dict)
        self.assertEqual(proxy_status.get("tool"), "agent_proxy_status")
        self.assertEqual(proxy_status.get("startup_trust"), "direct MCP proxy")
        self.assertEqual(proxy_status.get("resume_state"), "not applicable in direct MCP mode")

        runtime_status = self._manager().runtime_status()
        cached_proxy_status = runtime_status.get("proxy_status", {})
        self.assertIsInstance(cached_proxy_status, dict)
        self.assertEqual(cached_proxy_status.get("startup_trust"), "direct MCP proxy")
        self.assertIn(cached_proxy_status.get("runtime_status"), {"ready", "degraded"})

    async def test_tab_ownership_and_blank_cleanup(self):
        navigate_payload = await self._call_tool_json("browser_navigate", {"url": "https://example.com"})
        self.assertEqual(navigate_payload.get("tool"), "browser_navigate")
        self.assertIn(navigate_payload.get("status"), {"ok", "failed"})
        self.assertIn(int(navigate_payload.get("attempts", 0)), {1, 2})
        self.assertIn("verification", navigate_payload)

        tabs_new_payload = await self._call_tool_json("browser_tabs", {"action": "new"})
        self.assertEqual(tabs_new_payload.get("tool"), "browser_tabs")
        self.assertIn(tabs_new_payload.get("status"), {"ok", "failed"})

        close_payload = json.loads(await self._manager().browser_close_blank_tabs())
        self.assertEqual(close_payload.get("status"), "ok")
        urls_after_cleanup = [tab.get("url", "") for tab in close_payload.get("tabs", [])]
        self.assertTrue(urls_after_cleanup, "Expected at least one tab after cleanup")
        self.assertTrue(all(url != "about:blank" for url in urls_after_cleanup))

        navigate_verify_payload = await self._call_tool_json("browser_navigate", {"url": "https://www.example.org"})
        self.assertEqual(navigate_verify_payload.get("tool"), "browser_navigate")
        self.assertIn(navigate_verify_payload.get("status"), {"ok", "failed"})
        self.assertIn("verification", navigate_verify_payload)
        self.assertIn(int(navigate_verify_payload.get("attempts", 0)), {1, 2})

    async def test_retry_flow_for_retryable_click(self):
        payload = await self._call_tool_json("browser_click", {"element": "text:__definitely_not_present__"})
        self.assertEqual(payload.get("tool"), "browser_click")
        self.assertEqual(int(payload.get("attempts", 0)), 2)
        self.assertFalse(bool(payload.get("recovered", False)))
        self.assertIn(payload.get("status"), {"ok", "failed"})
        self.assertIn("verification", payload)

    async def test_non_retryable_tabs_select_stays_single_attempt(self):
        payload = await self._call_tool_json("browser_tabs", {"action": "select", "index": 9999})
        self.assertEqual(payload.get("tool"), "browser_tabs")
        self.assertEqual(int(payload.get("attempts", 0)), 1)
        self.assertIn(payload.get("status"), {"ok", "failed"})
        self.assertIn("verification", payload)


class MCPProxyLiveIntegrationTests(_BaseMCPLiveIntegrationTest):
    session: Optional[Any]
    exit_stack: Optional[AsyncExitStack]

    def setUp(self):
        super().setUp()
        self.session = None
        self.exit_stack = None
        self.output_dir = Path(os.environ.get("PLAYWRIGHT_MCP_OUTPUT_DIR", str(self.output_dir)))

    def _runtime_prefix(self) -> str:
        return "mcp-proxy-live"

    def _session(self) -> Any:
        self.assertIsNotNone(self.session, "Direct MCP session was not initialized")
        return self.session

    async def asyncSetUp(self):
        if not RUN_MCP_LIVE_TESTS:
            self.skipTest("Set RUN_MCP_LIVE_TESTS=1 to run live MCP integration tests.")

        from mcp import ClientSession, StdioServerParameters  # pyright: ignore[reportMissingImports]
        from mcp.client.stdio import stdio_client  # pyright: ignore[reportMissingImports]

        launcher = self.workspace_root / "playwright-edge-mcp.js"
        if not launcher.exists():
            self.fail(f"MCP launcher not found: {launcher}")

        env = os.environ.copy()
        env.update(
            {
                "PLAYWRIGHT_MCP_OWNER": env.get("PLAYWRIGHT_MCP_OWNER", "python-proxy"),
                "PLAYWRIGHT_MCP_FORCE_OWNER": "true",
                "PLAYWRIGHT_MCP_OWNER_FILE": env.get(
                    "PLAYWRIGHT_MCP_OWNER_FILE", str(self.workspace_root / ".playwright-mcp" / "active-owner.txt")
                ),
                "PLAYWRIGHT_MCP_USER_DATA_DIR": env.get(
                    "PLAYWRIGHT_MCP_USER_DATA_DIR", str(self.workspace_root / ".playwright-mcp" / "edge-profile")
                ),
                "PLAYWRIGHT_MCP_OUTPUT_DIR": env.get(
                    "PLAYWRIGHT_MCP_OUTPUT_DIR", str(self.workspace_root / ".playwright-mcp" / "output")
                ),
                "PLAYWRIGHT_MCP_SAVE_TRACE": env.get("PLAYWRIGHT_MCP_SAVE_TRACE", "true"),
                "PLAYWRIGHT_MCP_SAVE_SESSION": env.get("PLAYWRIGHT_MCP_SAVE_SESSION", "true"),
            }
        )

        server_params = StdioServerParameters(command="node", args=[str(launcher)], env=env)
        self.exit_stack = AsyncExitStack()
        try:
            read, write = await self.exit_stack.enter_async_context(stdio_client(server_params))
            self.session = await self.exit_stack.enter_async_context(ClientSession(read, write))
            await self._session().initialize()
        except Exception:
            self.exit_stack = None
            self.session = None
            self._restore_live_runtime()
            raise

    async def asyncTearDown(self):
        try:
            if self.exit_stack is not None:
                try:
                    await self.exit_stack.aclose()
                except Exception:
                    # Python 3.14 + anyio can raise a cancel-scope task mismatch during
                    # stdio_client teardown even after the session has been used successfully.
                    pass
            self.exit_stack = None
            self.session = None
        finally:
            self._restore_live_runtime()

    async def _list_tools(self):
        return await self._session().list_tools()

    async def _call_tool(self, tool_name: str, arguments: Optional[Dict[str, Any]] = None):
        return await self._session().call_tool(tool_name, arguments=arguments or {})

    async def test_tools_list_exposes_confirm_schema_fields(self):
        tools = await self._list_tools()
        browser_click = next((tool for tool in tools.tools if getattr(tool, "name", "") == "browser_click"), None)
        self.assertIsNotNone(browser_click, "browser_click tool missing from MCP tools/list")

        input_schema = getattr(browser_click, "inputSchema", {}) or {}
        properties = input_schema.get("properties", {})
        self.assertIn("confirm", properties)
        self.assertIn("confirm_token", properties)

    async def test_tools_list_exposes_proxy_status_tool(self):
        tools = await self._list_tools()
        status_tool = next((tool for tool in tools.tools if getattr(tool, "name", "") == "agent_proxy_status"), None)
        self.assertIsNotNone(status_tool, "agent_proxy_status tool missing from MCP tools/list")

    async def test_direct_proxy_status_reports_readiness_and_trust(self):
        result = await self._call_tool("agent_proxy_status", {})
        payload = getattr(result, "structuredContent", None)
        self.assertIsInstance(payload, dict)
        self.assertEqual(payload.get("tool"), "agent_proxy_status")
        self.assertIn(payload.get("runtime_status"), {"ready", "degraded"})
        self.assertEqual(payload.get("startup_trust"), "direct MCP proxy")
        self.assertEqual(payload.get("resume_state"), "not applicable in direct MCP mode")
        self.assertIn("ownership", payload)
        self.assertIn("safety", payload)

    async def test_direct_dangerous_click_requires_confirmation(self):
        result = await self._call_tool("browser_click", {"element": "text:Delete account"})
        payload = getattr(result, "structuredContent", None)
        self.assertIsInstance(payload, dict)
        self.assertEqual(payload.get("status"), "confirm_required")
        self.assertEqual(payload.get("tool"), "browser_click")
        self.assertTrue(str(payload.get("confirm_token", "")).strip())

    async def test_direct_navigation_includes_proxy_verification_and_evidence(self):
        result = await self._call_tool("browser_navigate", {"url": "https://example.com"})
        payload = getattr(result, "structuredContent", None)
        self.assertIsInstance(payload, dict)
        self.assertIn("safety", payload)
        self.assertIn("evidence", payload)
        self.assertIn("verification", payload)
        self.assertEqual(payload["safety"].get("tool"), "browser_navigate")
        self.assertIsInstance(payload["verification"].get("ok"), bool)
        self.assertIn(payload["evidence"].get("status"), {"verified", "verification_failed", "reported_ok", "failed"})

    async def test_direct_screenshot_includes_artifact_verification(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        screenshot_path = self.output_dir / "proxy-live-verification.png"
        if screenshot_path.exists():
            screenshot_path.unlink()

        await self._call_tool("browser_navigate", {"url": "https://example.com"})
        result = await self._call_tool("browser_take_screenshot", {"path": str(screenshot_path)})
        payload = getattr(result, "structuredContent", None)
        self.assertIsInstance(payload, dict)
        self.assertIn("verification", payload)
        output_path = str(payload["verification"]["details"].get("output_path", "")).strip()
        self.assertTrue(output_path, "Expected verification output_path to be present")
        resolved_output_path = Path(output_path)
        if not resolved_output_path.is_absolute():
            resolved_output_path = (self.workspace_root / resolved_output_path).resolve()
        self.assertTrue(resolved_output_path.exists(), f"Expected screenshot artifact to exist on disk: {resolved_output_path}")


if __name__ == "__main__":
    unittest.main()
