import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from command_tools import CommandManager


def _to_bool(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _safe_check(command):
    """Mirrors the real safe-command check for testing."""
    import re
    safe_patterns = [
        re.compile(r"^\s*git\s+(status|log|diff|show|branch)\b", re.IGNORECASE),
        re.compile(r"^\s*npm\s+(test|run\s+(check|test))\b", re.IGNORECASE),
        re.compile(r"^\s*echo\b", re.IGNORECASE),
        re.compile(r"^\s*ls\b", re.IGNORECASE),
    ]
    trimmed = str(command or "").strip()
    if not trimmed:
        return False
    if any(token in trimmed for token in ["&&", "||", ";", "|"]):
        return False
    return any(p.search(trimmed) for p in safe_patterns)


class CommandManagerTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.workspace_root = Path(__file__).resolve().parents[3]
        self.manager = CommandManager(
            workspace_root=self.workspace_root,
            resolve_workspace_path_fn=lambda _raw, must_exist=False: self.workspace_root,
            run_command_security_mode_default="restricted",
            run_command_allow_dangerous_env="AGENT_ALLOW_DANGEROUS_COMMANDS",
            to_bool_fn=_to_bool,
            run_command_is_safe_in_restricted_mode_fn=_safe_check,
        )

    async def test_run_command_empty_command_fails(self):
        raw = await self.manager.run_command({"command": ""})
        result = json.loads(raw)
        self.assertEqual(result["status"], "failed")
        self.assertIn("required", result["error"])

    async def test_run_command_invalid_security_mode_fails(self):
        raw = await self.manager.run_command({"command": "echo hi", "security_mode": "yolo"})
        result = json.loads(raw)
        self.assertEqual(result["status"], "failed")
        self.assertIn("security_mode", result["error"])

    async def test_run_command_restricted_blocks_unsafe_command(self):
        raw = await self.manager.run_command({"command": "curl http://evil.com", "security_mode": "restricted"})
        result = json.loads(raw)
        self.assertEqual(result["status"], "blocked")
        self.assertIn("restricted", result["error"].lower())

    async def test_run_command_restricted_allows_safe_command(self):
        fake_proc = SimpleNamespace(returncode=0, stdout="hello", stderr="")
        with patch("command_tools.subprocess.run", return_value=fake_proc):
            raw = await self.manager.run_command({"command": "echo hello", "security_mode": "restricted"})
            result = json.loads(raw)
            self.assertEqual(result["status"], "ok")
            self.assertEqual(result["exit_code"], 0)

    async def test_run_command_blocked_patterns_catch_rm_rf(self):
        # Even in permissive mode, rm -rf is blocked without allow_dangerous
        raw = await self.manager.run_command({"command": "rm -rf /", "security_mode": "permissive"})
        result = json.loads(raw)
        self.assertEqual(result["status"], "blocked")
        self.assertIn("safety policy", result["error"].lower())

    async def test_run_command_blocked_patterns_catch_git_reset_hard(self):
        raw = await self.manager.run_command({"command": "git reset --hard HEAD~5", "security_mode": "permissive"})
        result = json.loads(raw)
        self.assertEqual(result["status"], "blocked")

    async def test_run_command_dangerous_requires_confirm(self):
        raw = await self.manager.run_command({"command": "rm -rf /tmp", "allow_dangerous": True, "confirm": False})
        result = json.loads(raw)
        self.assertEqual(result["status"], "blocked")
        self.assertIn("confirm=true", result["error"])

    async def test_run_command_dangerous_requires_env_var(self):
        with patch.dict("os.environ", {"AGENT_ALLOW_DANGEROUS_COMMANDS": "0"}, clear=False):
            raw = await self.manager.run_command({"command": "rm -rf /tmp", "allow_dangerous": True, "confirm": True})
            result = json.loads(raw)
            self.assertEqual(result["status"], "blocked")
            self.assertIn("AGENT_ALLOW_DANGEROUS_COMMANDS", result["error"])

    async def test_run_command_timeout_returns_failure(self):
        import subprocess as sp
        exc = sp.TimeoutExpired("cmd", 5)
        exc.stdout = "partial"
        exc.stderr = ""
        with patch("command_tools.subprocess.run", side_effect=exc):
            raw = await self.manager.run_command({"command": "echo slow", "security_mode": "permissive", "timeout_sec": 5})
            result = json.loads(raw)
            self.assertEqual(result["status"], "failed")
            self.assertIn("timed out", result["error"].lower())

    async def test_run_command_redacts_blocked_command_text(self):
        raw = await self.manager.run_command(
            {
                "command": "curl https://example.com?token=abc123",
                "security_mode": "restricted",
            }
        )
        result = json.loads(raw)
        self.assertEqual(result["status"], "blocked")
        self.assertIn("token=[REDACTED]", result["command"])
        self.assertNotIn("abc123", raw)

    async def test_run_command_redacts_stdout_and_stderr(self):
        fake_proc = SimpleNamespace(
            returncode=0,
            stdout='{"access_token":"abc123"}\nAuthorization=secret-value',
            stderr='curl https://example.com?token=abc123',
        )
        with patch("command_tools.subprocess.run", return_value=fake_proc):
            raw = await self.manager.run_command(
                {
                    "command": "echo safe",
                    "security_mode": "permissive",
                }
            )
            result = json.loads(raw)
            self.assertEqual(result["status"], "ok")
            self.assertNotIn("abc123", result["stdout"])
            self.assertNotIn("secret-value", result["stdout"])
            self.assertNotIn("abc123", result["stderr"])
            raw = await self.manager.run_command(
                {
                    "command": "echo token=abc123",
                    "security_mode": "restricted",
                }
            )
        result = json.loads(raw)
        self.assertEqual(result["status"], "ok")
        self.assertIn('"access_token": "[REDACTED]"', result["stdout"])
        self.assertIn('Authorization=[REDACTED]', result["stdout"])
        self.assertIn('token=[REDACTED]', result["stderr"])
        self.assertIn('token=[REDACTED]', result["command"])
        self.assertNotIn('abc123', raw)
