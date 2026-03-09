import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import agent as agent_module
from tests_support import create_repo_local_temp_dir, remove_tree


class AgentPreflightTests(unittest.TestCase):
    def test_wants_mcp_status_mode_detects_flag(self):
        with patch.object(agent_module, "sys", SimpleNamespace(argv=["agent.py", "--mcp-status"])):
            self.assertTrue(agent_module._wants_mcp_status_mode())

    def test_wants_json_output_detects_flag(self):
        with patch.object(agent_module, "sys", SimpleNamespace(argv=["agent.py", "--json"])):
            self.assertTrue(agent_module._wants_json_output())

    def test_preflight_report_highlights_missing_env_and_mcp(self):
        fake_env = Path("C:/fake/.env")
        fake_example = Path(__file__).resolve()
        with (
            patch.object(agent_module, "ENV_FILE", fake_env),
            patch.object(agent_module, "ENV_EXAMPLE_FILE", fake_example),
            patch.object(agent_module, "client", None),
            patch.object(agent_module, "MODEL_PROVIDER", "openai"),
            patch.object(agent_module, "MODEL", "codex-5.3"),
            patch.object(agent_module, "MODEL_SETUP_ERROR", "OPENAI_API_KEY is missing."),
        ):
            report = agent_module._format_preflight_report(
                {
                    "status": "failed",
                    "connected": False,
                    "tool_count": 0,
                    "error": "Access is denied",
                }
            )
        self.assertIn("Agent Preflight", report)
        self.assertIn("model_ready: False", report)
        self.assertIn("mcp_status: failed", report)
        self.assertIn("copy .env.example .env", report)
        self.assertIn("npm run agent:vscode", report)

    def test_session_resume_guard_requires_model_ready_even_if_env_file_exists(self):
        fake_env = Path(__file__).resolve()
        with (
            patch.object(agent_module, "ENV_FILE", fake_env),
            patch.object(agent_module, "client", None),
        ):
            result = agent_module._session_resume_guard({"startup_completed": True})
        self.assertFalse(result["allowed"])
        self.assertFalse(result["model_ready"])
        self.assertTrue(result["env_file_exists"])
        self.assertEqual(result["reason"], "local model config is not ready yet")

    def test_session_resume_guard_allows_after_successful_startup_with_model_ready(self):
        fake_client = object()
        with patch.object(agent_module, "client", fake_client):
            result = agent_module._session_resume_guard({"startup_completed": True})
        self.assertTrue(result["allowed"])
        self.assertTrue(result["model_ready"])
        self.assertEqual(result["reason"], "")

    def test_runtime_status_reports_browser_tool_usability_and_resume_fields(self):
        fake_env = Path(__file__).resolve()
        fake_example = Path(__file__).resolve()
        fake_client = object()
        fake_functions = {
            "browser_click": object(),
            "browser_navigate": object(),
            "fs_read": object(),
        }
        with (
            patch.object(agent_module, "ENV_FILE", fake_env),
            patch.object(agent_module, "ENV_EXAMPLE_FILE", fake_example),
            patch.object(agent_module, "client", fake_client),
            patch.object(agent_module, "AVAILABLE_FUNCTIONS", fake_functions),
            patch.object(
                agent_module,
                "get_mcp_runtime_status",
                return_value={
                    "status": "connected",
                    "connected": True,
                    "tool_count": 12,
                    "error": "",
                    "proxy_status": {
                        "runtime_status": "ready",
                        "startup_trust": "direct MCP proxy",
                        "resume_state": "not applicable in direct MCP mode",
                        "summary": "Guarded direct MCP mode is ready; resume state is not applicable in direct MCP mode.",
                        "ownership": {"owner": "python", "active_owner": "python"},
                    },
                },
            ),
            patch.object(
                agent_module,
                "_load_startup_state",
                return_value={
                    "startup_completed": True,
                    "startup_completion_reason": "direct_answer",
                    "startup_used_tools": False,
                    "successful_startup_at": "2026-03-07T10:00:00",
                    "successful_startup_count": 1,
                },
            ),
        ):
            status = agent_module._format_runtime_status([])
        self.assertIn("model_ready: True", status)
        self.assertIn("env_file:", status)
        self.assertIn("browser_tools_registered: 2", status)
        self.assertIn("browser_tools_usable: True", status)
        self.assertIn("startup_completion_reason: direct_answer", status)
        self.assertIn("startup_used_tools: False", status)
        self.assertIn("startup_trust: direct answer", status)
        self.assertIn("resume_state: ready for auto-resume", status)
        self.assertIn("session_auto_resume_ready: True", status)
        self.assertIn("mcp_proxy_runtime_status: ready", status)
        self.assertIn("mcp_proxy_startup_trust: direct MCP proxy", status)
        self.assertIn("mcp_proxy_resume_state: not applicable in direct MCP mode", status)
        self.assertIn("mcp_proxy_owner: python (active=python)", status)

    def test_preflight_marks_local_env_copy_as_optional_when_model_is_already_ready(self):
        fake_env = Path("C:/fake/.env")
        fake_example = Path(__file__).resolve()
        fake_client = object()
        with (
            patch.object(agent_module, "ENV_FILE", fake_env),
            patch.object(agent_module, "ENV_EXAMPLE_FILE", fake_example),
            patch.object(agent_module, "client", fake_client),
            patch.object(agent_module, "_load_startup_state", return_value={"startup_completed": False}),
        ):
            report = agent_module._format_preflight_report(
                {
                    "status": "failed",
                    "connected": False,
                    "tool_count": 0,
                    "error": "Access is denied",
                }
            )
        self.assertIn("optional local config: copy .env.example .env", report)
        self.assertNotIn("create local config: copy .env.example .env", report)
        self.assertIn("resume_state: locked until first successful turn", report)
        self.assertIn("startup_completion_reason: none", report)
        self.assertIn("startup_used_tools: none", report)
        self.assertIn("startup_trust: not yet earned", report)

    def test_preflight_report_includes_mcp_proxy_summary_when_connected(self):
        fake_env = Path(__file__).resolve()
        fake_example = Path(__file__).resolve()
        fake_client = object()
        with (
            patch.object(agent_module, "ENV_FILE", fake_env),
            patch.object(agent_module, "ENV_EXAMPLE_FILE", fake_example),
            patch.object(agent_module, "client", fake_client),
            patch.object(agent_module, "_load_startup_state", return_value={"startup_completed": False}),
        ):
            report = agent_module._format_preflight_report(
                {
                    "status": "connected",
                    "connected": True,
                    "tool_count": 13,
                    "error": "",
                    "proxy_status": {
                        "runtime_status": "ready",
                        "startup_trust": "direct MCP proxy",
                        "resume_state": "not applicable in direct MCP mode",
                        "summary": "Guarded direct MCP mode is ready; resume state is not applicable in direct MCP mode.",
                        "ownership": {"owner": "python", "active_owner": "python"},
                    },
                }
            )
        self.assertIn("mcp_proxy_runtime_status: ready", report)
        self.assertIn("mcp_proxy_startup_trust: direct MCP proxy", report)
        self.assertIn("mcp_proxy_resume_state: not applicable in direct MCP mode", report)

    def test_mcp_startup_note_includes_proxy_summary_when_connected(self):
        note = agent_module._format_mcp_startup_note(
            {
                "connected": True,
                "tool_count": 13,
                "proxy_status": {
                    "summary": "Guarded direct MCP mode is ready; resume state is not applicable in direct MCP mode.",
                },
            }
        )
        self.assertIn("[MCP] Proxy: Guarded direct MCP mode is ready; resume state is not applicable in direct MCP mode.", note)

    def test_help_text_mentions_mcp_command(self):
        help_text = agent_module._help_text()
        self.assertIn("/mcp", help_text)

    def test_build_turn_tool_context_uses_routed_subset(self):
        fake_tools = [
            {"type": "function", "function": {"name": "calculate", "description": "", "parameters": {"type": "object", "properties": {}, "required": []}}},
            {"type": "function", "function": {"name": "reasoning_plan", "description": "", "parameters": {"type": "object", "properties": {}, "required": []}}},
            {"type": "function", "function": {"name": "fs_read", "description": "", "parameters": {"type": "object", "properties": {}, "required": []}}},
            {"type": "function", "function": {"name": "browser_click", "description": "", "parameters": {"type": "object", "properties": {}, "required": []}}},
            {"type": "function", "function": {"name": "call_tool", "description": "", "parameters": {"type": "object", "properties": {}, "required": []}}},
        ]
        fake_functions = {
            "calculate": object(),
            "reasoning_plan": object(),
            "fs_read": object(),
            "browser_click": object(),
            "call_tool": object(),
        }
        with (
            patch.object(agent_module, "AGENT_TOOLS", fake_tools),
            patch.object(agent_module, "AVAILABLE_FUNCTIONS", fake_functions),
        ):
            context = agent_module._build_turn_tool_context(
                "Read this repo and inspect the local files.",
                mcp_status={"connected": True},
            )
        self.assertIn("fs_read", context["allowed_tool_names"])
        self.assertNotIn("browser_click", context["allowed_tool_names"])
        self.assertNotIn("call_tool", context["allowed_tool_names"])

    def test_startup_banner_mentions_mcp_command(self):
        banner = agent_module._format_startup_banner()
        self.assertIn("/mcp", banner)

    def test_format_mcp_proxy_status_report_includes_proxy_summary(self):
        report = agent_module._format_mcp_proxy_status_report(
            {
                "status": "connected",
                "connected": True,
                "tool_count": 13,
                "error": "",
                "proxy_status": {
                    "runtime_status": "ready",
                    "startup_trust": "direct MCP proxy",
                    "resume_state": "not applicable in direct MCP mode",
                    "summary": "Guarded direct MCP mode is ready; resume state is not applicable in direct MCP mode.",
                    "ownership": {"owner": "python", "active_owner": "python"},
                },
            }
        )
        self.assertIn("MCP Proxy Status", report)
        self.assertIn("mcp_proxy_runtime_status: ready", report)
        self.assertIn("mcp_proxy_summary: Guarded direct MCP mode is ready; resume state is not applicable in direct MCP mode.", report)

    def test_build_mcp_proxy_status_payload_returns_structured_report(self):
        payload = agent_module._build_mcp_proxy_status_payload(
            {
                "status": "connected",
                "connected": True,
                "tool_count": 13,
                "error": "",
                "proxy_status": {
                    "runtime_status": "ready",
                    "startup_trust": "direct MCP proxy",
                    "resume_state": "not applicable in direct MCP mode",
                    "summary": "Guarded direct MCP mode is ready.",
                },
            }
        )
        self.assertEqual(payload["report"], "mcp_proxy_status")
        self.assertTrue(payload["mcp"]["connected"])
        self.assertEqual(payload["proxy"]["runtime_status"], "ready")

    def test_mark_successful_startup_records_completion_reason(self):
        fake_status = {"connected": True, "tool_count": 3}
        with (
            patch.object(agent_module, "_load_startup_state", return_value={}),
            patch.object(agent_module, "_save_startup_state") as save_startup_state,
            patch.object(agent_module, "get_mcp_runtime_status", return_value=fake_status),
        ):
            agent_module._mark_successful_startup(reason="tool_assisted_answer")
        save_startup_state.assert_called_once()
        payload = save_startup_state.call_args.args[0]
        self.assertTrue(payload["startup_completed"])
        self.assertEqual(payload["startup_completion_reason"], "tool_assisted_answer")
        self.assertTrue(payload["startup_used_tools"])
        self.assertEqual(payload["mcp_tool_count"], 3)

    def test_describe_resume_state_normalizes_internal_no_resume_reason(self):
        summary = agent_module._describe_resume_state(
            {"allowed": False, "reason": "no resume"},
            {"startup_completed": False},
        )
        self.assertEqual(summary, "locked until first successful turn")

    def test_trusted_startup_reason_requires_successful_tool_results(self):
        trusted = agent_module._trusted_startup_reason_for_turn(
            True,
            ['{"status":"ok"}', {"status": "success"}],
        )
        untrusted = agent_module._trusted_startup_reason_for_turn(
            True,
            ['{"status":"preview_required"}'],
        )
        self.assertEqual(trusted, "tool_assisted_answer")
        self.assertIsNone(untrusted)

    def test_describe_startup_trust_for_tool_assisted_answer(self):
        summary = agent_module._describe_startup_trust(
            {
                "startup_completed": True,
                "startup_completion_reason": "tool_assisted_answer",
                "startup_used_tools": True,
            }
        )
        self.assertEqual(summary, "tool-assisted verified")


class AgentSessionPersistenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp_path = create_repo_local_temp_dir(Path(__file__), "test-agent-preflight", "session-state")

    def tearDown(self):
        remove_tree(self.tmp_path)

    def test_save_session_state_skips_untrusted_run(self):
        session_file = self.tmp_path / "last_session.json"
        with patch.object(agent_module, "SESSION_STATE_FILE", session_file):
            wrote = agent_module._save_session_state(
                [{"role": "user", "content": "hello"}],
                session_trusted=False,
            )
        self.assertFalse(wrote)
        self.assertFalse(session_file.exists())

    def test_save_session_state_writes_for_trusted_run(self):
        session_file = self.tmp_path / "last_session.json"
        with patch.object(agent_module, "SESSION_STATE_FILE", session_file):
            wrote = agent_module._save_session_state(
                [{"role": "user", "content": "hello"}],
                session_trusted=True,
            )
        self.assertTrue(wrote)
        self.assertTrue(session_file.exists())


class _FakeChatCompletions:
    def __init__(self, responses):
        self._responses = list(responses)

    def create(self, **kwargs):
        if not self._responses:
            raise AssertionError("No fake responses left for client.chat.completions.create")
        return self._responses.pop(0)


class _FakeClient:
    def __init__(self, responses):
        self.chat = SimpleNamespace(completions=_FakeChatCompletions(responses))


class AgentRunLoopTests(unittest.IsolatedAsyncioTestCase):
    async def test_main_runs_mcp_status_mode_without_model_client(self):
        with (
            patch.object(agent_module, "_wants_preflight_mode", return_value=False),
            patch.object(agent_module, "_wants_mcp_status_mode", return_value=True),
            patch.object(agent_module, "_wants_json_output", return_value=False),
            patch.object(agent_module, "init_mcp_client", AsyncMock(return_value={"connected": True, "status": "connected"})),
            patch.object(agent_module, "shutdown_mcp_client", AsyncMock()) as shutdown_mcp_client,
            patch.object(agent_module, "_format_mcp_proxy_status_report", return_value="MCP Proxy Status\n- mcp_connected: True"),
            patch.object(agent_module, "run_agent", AsyncMock()) as run_agent,
            patch.object(agent_module, "client", None),
            patch("builtins.print") as print_mock,
        ):
            await agent_module.main()
        run_agent.assert_not_called()
        shutdown_mcp_client.assert_awaited_once()
        rendered = "\n".join(" ".join(str(arg) for arg in call.args) for call in print_mock.call_args_list if call.args)
        self.assertIn("MCP Proxy Status", rendered)

    async def test_main_runs_mcp_status_mode_as_json_without_model_client(self):
        with (
            patch.object(agent_module, "_wants_preflight_mode", return_value=False),
            patch.object(agent_module, "_wants_mcp_status_mode", return_value=True),
            patch.object(agent_module, "_wants_json_output", return_value=True),
            patch.object(agent_module, "init_mcp_client", AsyncMock(return_value={"connected": False, "status": "failed", "error": "Access is denied"})),
            patch.object(agent_module, "shutdown_mcp_client", AsyncMock()) as shutdown_mcp_client,
            patch.object(
                agent_module,
                "_build_mcp_proxy_status_payload",
                return_value={"report": "mcp_proxy_status", "mcp": {"status": "failed"}, "proxy": {"runtime_status": "unavailable"}},
            ),
            patch.object(agent_module, "run_agent", AsyncMock()) as run_agent,
            patch.object(agent_module, "client", None),
            patch("builtins.print") as print_mock,
        ):
            await agent_module.main()
        run_agent.assert_not_called()
        shutdown_mcp_client.assert_awaited_once()
        rendered = "\n".join(" ".join(str(arg) for arg in call.args) for call in print_mock.call_args_list if call.args)
        self.assertIn('"report": "mcp_proxy_status"', rendered)

    async def test_run_agent_does_not_mark_startup_on_immediate_quit(self):
        with (
            patch.object(agent_module, "_build_base_messages", AsyncMock(return_value=([], None))),
            patch.object(agent_module, "_session_resume_guard", return_value={"allowed": False, "reason": "no resume"}),
            patch.object(agent_module, "_save_session_state") as save_session_state,
            patch.object(agent_module, "_mark_successful_startup") as mark_startup,
            patch.object(agent_module, "_memory_log_event", AsyncMock()),
            patch.object(agent_module, "_preflight_local_context", AsyncMock(return_value=None)),
            patch.object(agent_module, "_memory_recall_context", AsyncMock(return_value=None)),
            patch("builtins.input", side_effect=["quit"]),
        ):
            await agent_module.run_agent()
        mark_startup.assert_not_called()
        save_session_state.assert_called_once_with([], session_trusted=False)

    async def test_run_agent_marks_startup_after_first_completed_turn(self):
        fake_response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="done",
                        tool_calls=None,
                    )
                )
            ]
        )
        fake_client = _FakeClient([fake_response])
        with (
            patch.object(agent_module, "_build_base_messages", AsyncMock(return_value=([], None))),
            patch.object(agent_module, "_session_resume_guard", return_value={"allowed": False, "reason": "no resume"}),
            patch.object(agent_module, "_save_session_state") as save_session_state,
            patch.object(agent_module, "_mark_successful_startup") as mark_startup,
            patch.object(agent_module, "_memory_log_event", AsyncMock()),
            patch.object(agent_module, "_preflight_local_context", AsyncMock(return_value=None)),
            patch.object(agent_module, "_memory_recall_context", AsyncMock(return_value=None)),
            patch.object(agent_module, "client", fake_client),
            patch("builtins.input", side_effect=["hello", "quit"]),
        ):
            await agent_module.run_agent()
        mark_startup.assert_called_once_with(reason="direct_answer")
        save_session_state.assert_called_once()
        self.assertTrue(save_session_state.call_args.kwargs["session_trusted"])

    async def test_run_agent_marks_tool_assisted_startup_after_verified_answer(self):
        fake_tool_call = SimpleNamespace(
            id="call_1",
            function=SimpleNamespace(name="fs_read", arguments="{}"),
        )
        tool_response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="",
                        tool_calls=[fake_tool_call],
                    )
                )
            ]
        )
        final_response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="done with tools",
                        tool_calls=None,
                    )
                )
            ]
        )
        fake_client = _FakeClient([tool_response, final_response])
        with (
            patch.object(agent_module, "_build_base_messages", AsyncMock(return_value=([], None))),
            patch.object(agent_module, "_session_resume_guard", return_value={"allowed": False, "reason": "no resume"}),
            patch.object(agent_module, "_save_session_state") as save_session_state,
            patch.object(agent_module, "_mark_successful_startup") as mark_startup,
            patch.object(agent_module, "_memory_log_event", AsyncMock()),
            patch.object(agent_module, "_preflight_local_context", AsyncMock(return_value=None)),
            patch.object(agent_module, "_memory_recall_context", AsyncMock(return_value=None)),
            patch.object(agent_module, "_execute_tool_call", AsyncMock(return_value='{"status":"ok"}')),
            patch.object(agent_module, "client", fake_client),
            patch("builtins.input", side_effect=["hello", "quit"]),
        ):
            await agent_module.run_agent()
        mark_startup.assert_called_once_with(reason="tool_assisted_answer")
        save_session_state.assert_called_once()
        self.assertTrue(save_session_state.call_args.kwargs["session_trusted"])

    async def test_run_agent_keeps_startup_locked_when_tool_result_is_not_successful(self):
        fake_tool_call = SimpleNamespace(
            id="call_1",
            function=SimpleNamespace(name="fs_read", arguments="{}"),
        )
        tool_response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="",
                        tool_calls=[fake_tool_call],
                    )
                )
            ]
        )
        final_response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="done but tool was gated",
                        tool_calls=None,
                    )
                )
            ]
        )
        fake_client = _FakeClient([tool_response, final_response])
        with (
            patch.object(agent_module, "_build_base_messages", AsyncMock(return_value=([], None))),
            patch.object(agent_module, "_session_resume_guard", return_value={"allowed": False, "reason": "no resume"}),
            patch.object(agent_module, "_save_session_state") as save_session_state,
            patch.object(agent_module, "_mark_successful_startup") as mark_startup,
            patch.object(agent_module, "_memory_log_event", AsyncMock()),
            patch.object(agent_module, "_preflight_local_context", AsyncMock(return_value=None)),
            patch.object(agent_module, "_memory_recall_context", AsyncMock(return_value=None)),
            patch.object(agent_module, "_execute_tool_call", AsyncMock(return_value='{"status":"preview_required"}')),
            patch.object(agent_module, "client", fake_client),
            patch("builtins.input", side_effect=["hello", "quit"]),
        ):
            await agent_module.run_agent()
        mark_startup.assert_not_called()
        save_session_state.assert_called_once()
        self.assertFalse(save_session_state.call_args.kwargs["session_trusted"])

    async def test_run_agent_does_not_mark_startup_on_timeout_only_turn(self):
        fake_tool_call = SimpleNamespace(
            id="call_1",
            function=SimpleNamespace(name="fs_read", arguments="{}"),
        )
        fake_response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="",
                        tool_calls=[fake_tool_call],
                    )
                )
            ]
        )
        fake_client = _FakeClient([fake_response])
        with (
            patch.object(agent_module, "_build_base_messages", AsyncMock(return_value=([], None))),
            patch.object(agent_module, "_session_resume_guard", return_value={"allowed": False, "reason": "no resume"}),
            patch.object(agent_module, "_save_session_state") as save_session_state,
            patch.object(agent_module, "_mark_successful_startup") as mark_startup,
            patch.object(agent_module, "_memory_log_event", AsyncMock()),
            patch.object(agent_module, "_preflight_local_context", AsyncMock(return_value=None)),
            patch.object(agent_module, "_memory_recall_context", AsyncMock(return_value=None)),
            patch.object(agent_module, "_execute_tool_call", AsyncMock(return_value='{"status":"ok"}')),
            patch.object(agent_module, "client", fake_client),
            patch.object(agent_module, "MAX_ITERATIONS", 1),
            patch("builtins.input", side_effect=["loop", "quit"]),
        ):
            await agent_module.run_agent()
        mark_startup.assert_not_called()
        save_session_state.assert_called_once()
        self.assertFalse(save_session_state.call_args.kwargs["session_trusted"])

    async def test_run_agent_reports_resume_summary_when_save_is_skipped(self):
        with (
            patch.object(agent_module, "_build_base_messages", AsyncMock(return_value=([], None))),
            patch.object(agent_module, "_session_resume_guard", return_value={"allowed": False, "reason": "no resume"}),
            patch.object(agent_module, "_save_session_state", return_value=False),
            patch.object(agent_module, "_memory_log_event", AsyncMock()),
            patch.object(agent_module, "_preflight_local_context", AsyncMock(return_value=None)),
            patch.object(agent_module, "_memory_recall_context", AsyncMock(return_value=None)),
            patch("builtins.print") as print_mock,
            patch("builtins.input", side_effect=["/save", "quit"]),
        ):
            await agent_module.run_agent()
        rendered = "\n".join(" ".join(str(arg) for arg in call.args) for call in print_mock.call_args_list if call.args)
        self.assertIn("Checkpoint skipped: resume state is locked until first successful turn.", rendered)

    async def test_run_agent_prints_resume_summary_on_startup(self):
        with (
            patch.object(agent_module, "_build_base_messages", AsyncMock(return_value=([], None))),
            patch.object(
                agent_module,
                "_session_resume_guard",
                return_value={
                    "allowed": False,
                    "reason": "complete one successful startup before auto-resume",
                    "startup_state": {"startup_completed": False},
                },
            ),
            patch.object(agent_module, "_save_session_state"),
            patch.object(agent_module, "_memory_log_event", AsyncMock()),
            patch.object(agent_module, "_preflight_local_context", AsyncMock(return_value=None)),
            patch.object(agent_module, "_memory_recall_context", AsyncMock(return_value=None)),
            patch("builtins.print") as print_mock,
            patch("builtins.input", side_effect=["quit"]),
        ):
            await agent_module.run_agent()
        rendered = "\n".join(" ".join(str(arg) for arg in call.args) for call in print_mock.call_args_list if call.args)
        self.assertIn("[Session] Resume state: locked until first successful turn.", rendered)

    async def test_run_agent_normalizes_auto_resume_skipped_message(self):
        tmp_path = create_repo_local_temp_dir(Path(__file__), "test-agent-preflight", "resume-skip")
        try:
            session_file = tmp_path / "last_session.json"
            session_file.write_text('{"messages":[]}', encoding="utf-8")
            with (
                patch.object(agent_module, "_build_base_messages", AsyncMock(return_value=([], None))),
                patch.object(
                    agent_module,
                    "_session_resume_guard",
                    return_value={
                        "allowed": False,
                        "reason": "no resume",
                        "startup_state": {"startup_completed": False},
                    },
                ),
                patch.object(agent_module, "SESSION_STATE_FILE", session_file),
                patch.object(agent_module, "_save_session_state"),
                patch.object(agent_module, "_memory_log_event", AsyncMock()),
                patch.object(agent_module, "_preflight_local_context", AsyncMock(return_value=None)),
                patch.object(agent_module, "_memory_recall_context", AsyncMock(return_value=None)),
                patch("builtins.print") as print_mock,
                patch("builtins.input", side_effect=["quit"]),
            ):
                await agent_module.run_agent()
            rendered = "\n".join(" ".join(str(arg) for arg in call.args) for call in print_mock.call_args_list if call.args)
            self.assertIn("[Session] Auto-resume skipped: locked until first successful turn.", rendered)
        finally:
            remove_tree(tmp_path)


if __name__ == "__main__":
    unittest.main()
