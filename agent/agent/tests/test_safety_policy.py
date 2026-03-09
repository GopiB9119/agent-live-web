import json
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from architecture.safety_confirm import issue_confirm_token, validate_confirm_token
from architecture.safety_policy import evaluate_tool_call
from tooling.registry import ensure_safety_confirm_schema_fields
from tools import execute_tool_with_policy


class SafetyPolicyTests(unittest.TestCase):
    def setUp(self):
        self.workspace = Path(__file__).resolve().parents[3]

    def test_fs_patch_allows_scoped_write_with_verification(self):
        decision = evaluate_tool_call(
            "fs_patch",
            {
                "path": "README.md",
                "edits": [{"find": "Agent Live Web", "replace": "Agent Live Web"}],
            },
            self.workspace,
        )
        self.assertEqual(decision.decision, "allow_with_verification")
        self.assertEqual(decision.action_class, "scoped_reversible_write")

    def test_fs_write_existing_file_requires_confirmation(self):
        decision = evaluate_tool_call(
            "fs_write",
            {
                "path": "README.md",
                "content": "new content",
            },
            self.workspace,
        )
        self.assertEqual(decision.decision, "confirm_required")
        self.assertTrue(decision.confirm_token)

    def test_run_command_permissive_requires_preview(self):
        decision = evaluate_tool_call(
            "run_command",
            {
                "command": "npm run build",
                "security_mode": "permissive",
            },
            self.workspace,
        )
        self.assertEqual(decision.decision, "preview_required")

    def test_browser_dangerous_intent_requires_confirmation(self):
        decision = evaluate_tool_call(
            "browser_click",
            {
                "element": "text:Delete account",
            },
            self.workspace,
        )
        self.assertEqual(decision.decision, "confirm_required")

    def test_destructive_delete_blocked_without_operator_mode(self):
        decision = evaluate_tool_call(
            "fs_delete",
            {
                "path": "README.md",
            },
            self.workspace,
        )
        self.assertEqual(decision.decision, "blocked")

    def test_confirm_token_round_trip(self):
        args = {"path": "README.md", "content": "abc"}
        token = issue_confirm_token("fs_write", args)
        self.assertTrue(validate_confirm_token("fs_write", args, token))
        self.assertFalse(validate_confirm_token("fs_write", {"path": "README.md", "content": "changed"}, token))

    def test_registry_adds_confirm_fields(self):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "alpha",
                    "description": "alpha",
                    "parameters": {"type": "object", "properties": {}, "required": []},
                },
            }
        ]
        ensure_safety_confirm_schema_fields(tools)
        props = tools[0]["function"]["parameters"]["properties"]
        self.assertIn("confirm", props)
        self.assertIn("confirm_token", props)

    def test_workflow_execute_preview_required_when_steps_are_stateful(self):
        decision = evaluate_tool_call(
            "workflow_execute",
            {
                "steps": [
                    {"tool_name": "fs_read", "arguments": {"path": "README.md"}},
                    {"tool_name": "fs_write", "arguments": {"path": "tmp.txt", "content": "abc"}},
                ]
            },
            self.workspace,
        )
        self.assertEqual(decision.decision, "preview_required")


class GuardedExecutionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.workspace = Path(__file__).resolve().parents[3]
        self.workspace_tmp = self.workspace / ".tmp"
        self.workspace_tmp.mkdir(parents=True, exist_ok=True)
        self.audit_dir = self.workspace_tmp / "test-safety-policy"
        self.audit_dir.mkdir(parents=True, exist_ok=True)
        self.audit_file = self.audit_dir / "safety-events.jsonl"
        self.prev_audit_file = os.environ.get("AGENT_SAFETY_AUDIT_FILE")
        self.prev_audit_enabled = os.environ.get("AGENT_SAFETY_AUDIT_ENABLED")
        os.environ["AGENT_SAFETY_AUDIT_FILE"] = str(self.audit_file)
        os.environ["AGENT_SAFETY_AUDIT_ENABLED"] = "1"
        self.generated_file = self.workspace / ".tmp" / "safety-policy-generated.txt"
        if self.audit_file.exists():
            self.audit_file.unlink()
        if self.generated_file.exists():
            self.generated_file.unlink()

    def tearDown(self):
        if self.prev_audit_file is None:
            os.environ.pop("AGENT_SAFETY_AUDIT_FILE", None)
        else:
            os.environ["AGENT_SAFETY_AUDIT_FILE"] = self.prev_audit_file
        if self.prev_audit_enabled is None:
            os.environ.pop("AGENT_SAFETY_AUDIT_ENABLED", None)
        else:
            os.environ["AGENT_SAFETY_AUDIT_ENABLED"] = self.prev_audit_enabled
        if self.generated_file.exists():
            self.generated_file.unlink()
        if self.audit_file.exists():
            self.audit_file.unlink()

    def _read_audit_lines(self):
        if not self.audit_file.exists():
            return []
        return [json.loads(line) for line in self.audit_file.read_text(encoding="utf-8").splitlines() if line.strip()]

    async def test_preview_required_response_includes_preview_and_audit(self):
        raw = await execute_tool_with_policy(
            "run_command",
            {
                "command": "npm run build",
                "security_mode": "permissive",
            },
        )
        payload = json.loads(raw)
        self.assertEqual(payload["status"], "preview_required")
        self.assertIn("preview", payload)
        self.assertEqual(payload["preview"]["status"], "ok")
        self.assertEqual(payload["preview"]["preview"]["security_mode"], "permissive")

        audit_lines = self._read_audit_lines()
        self.assertTrue(any(line.get("event_type") == "decision" and line.get("tool") == "run_command" for line in audit_lines))

    async def test_confirm_token_allows_new_file_write_and_logs_execution(self):
        relative_path = self.generated_file.relative_to(self.workspace).as_posix()
        first = json.loads(
            await execute_tool_with_policy(
                "fs_write",
                {
                    "path": relative_path,
                    "content": "generated for safety gate test",
                },
            )
        )
        self.assertEqual(first["status"], "preview_required")
        token = first["confirm_token"]

        second = json.loads(
            await execute_tool_with_policy(
                "fs_write",
                {
                    "path": relative_path,
                    "content": "generated for safety gate test",
                    "confirm": True,
                    "confirm_token": token,
                },
            )
        )
        self.assertEqual(second["status"], "ok")
        self.assertTrue(self.generated_file.exists())

        audit_lines = self._read_audit_lines()
        self.assertTrue(any(line.get("event_type") == "decision" and line.get("tool") == "fs_write" for line in audit_lines))
        self.assertTrue(any(line.get("event_type") == "execution" and line.get("tool") == "fs_write" for line in audit_lines))


if __name__ == "__main__":
    unittest.main()
