import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import runtime_utils as ru


class RuntimeUtilsTests(unittest.TestCase):
    def test_to_bool(self):
        self.assertTrue(ru.to_bool("true"))
        self.assertTrue(ru.to_bool("1"))
        self.assertFalse(ru.to_bool("false"))
        self.assertFalse(ru.to_bool(None, default=False))
        self.assertTrue(ru.to_bool(None, default=True))

    def test_run_command_is_safe_in_restricted_mode(self):
        self.assertTrue(ru.run_command_is_safe_in_restricted_mode("git status"))
        self.assertFalse(ru.run_command_is_safe_in_restricted_mode("git status && whoami"))
        self.assertFalse(ru.run_command_is_safe_in_restricted_mode(""))

    def test_redact_sensitive_text(self):
        sample = "Authorization=abc123 token=my-secret-value Bearer qwertyuiopasdfgh123456"
        redacted = ru.redact_sensitive_text(sample)
        self.assertIn("Bearer [REDACTED]", redacted)
        self.assertIn("token=[REDACTED]", redacted)
        self.assertNotIn("my-secret-value", redacted)

    def test_resolve_workspace_path(self):
        resolved = ru.resolve_workspace_path("README.md", must_exist=False)
        self.assertTrue(isinstance(resolved, Path))
        self.assertTrue(str(resolved).startswith(str(ru.WORKSPACE_ROOT)))

    def test_resolve_workspace_path_rejects_outside(self):
        outside = Path(ru.WORKSPACE_ROOT.anchor) / "Windows"
        if not str(outside).startswith(str(ru.WORKSPACE_ROOT)):
            with self.assertRaises(ValueError):
                ru.resolve_workspace_path(str(outside), must_exist=False)

    def test_is_private_or_local_host(self):
        self.assertTrue(ru.is_private_or_local_host("localhost"))
        self.assertTrue(ru.is_private_or_local_host("127.0.0.1"))
        # Public documentation example domain should not be private/local.
        self.assertFalse(ru.is_private_or_local_host("example.com"))


if __name__ == "__main__":
    unittest.main()
