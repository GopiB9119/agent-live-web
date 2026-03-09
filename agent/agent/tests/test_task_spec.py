import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from task_spec import build_task_spec


class TaskSpecTests(unittest.TestCase):
    def test_build_task_spec_infers_hybrid_browser_workspace_task(self):
        spec = build_task_spec(
            "Open the admin website, update the user status, and verify the repo test file without submit."
        )
        self.assertEqual(spec["primary_kind"], "hybrid")
        self.assertEqual(spec["task_mode"], "test")
        self.assertTrue(spec["no_submit"])
        self.assertIn("browser", spec["surfaces_hint"])
        self.assertIn("workspace", spec["surfaces_hint"])
        self.assertIn("submit", spec["requested_actions"])

    def test_build_task_spec_normalizes_structured_payload(self):
        spec = build_task_spec(
            {
                "task": "Update a user in staging",
                "mode": "dry_run",
                "workspace": {"target_files": ["tests/admin.spec.ts"]},
                "website": {"base_url": "https://admin.example.com", "environment": "staging"},
                "operations": [{"action": "navigate", "target": "/users"}],
                "validation": {"must_not_submit_without_confirm": True},
                "safety": {"allow_write": True, "allow_submit": False, "data_sensitivity": "internal"},
            }
        )
        self.assertEqual(spec["source"], "structured")
        self.assertEqual(spec["environment"], "staging")
        self.assertEqual(spec["primary_kind"], "hybrid")
        self.assertEqual(spec["task_mode"], "mutate")
        self.assertTrue(spec["no_submit"])
        self.assertEqual(spec["data_sensitivity"], "internal")
        self.assertEqual(spec["risk_level"], "medium")


if __name__ == "__main__":
    unittest.main()
