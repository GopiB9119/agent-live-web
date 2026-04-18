import json
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

    def test_redact_sensitive_text_redacts_sensitive_url_parts(self):
        sample = "https://demo:secret@example.com/callback?token=abc123&state=ok"
        redacted = ru.redact_sensitive_text(sample)
        self.assertIn("https://[REDACTED]@example.com", redacted)
        self.assertIn("token=[REDACTED]", redacted)
        self.assertNotIn("abc123", redacted)

    def test_redact_sensitive_data_redacts_nested_secret_fields(self):
        sample = {
            "headers": {"Authorization": "Bearer abcdefghijklmnopqrstuvwxyz"},
            "request": {
                "url": "https://example.com/api?access_token=abc123&state=ok",
                "token": "plain-secret",
                "items": [
                    {"cookie": "sessionid=123"},
                    {"safe": "value"},
                ],
            },
        }

        redacted = ru.redact_sensitive_data(sample)
        self.assertEqual(redacted["headers"]["Authorization"], "[REDACTED]")
        self.assertEqual(redacted["request"]["token"], "[REDACTED]")
        self.assertIn("access_token=[REDACTED]", redacted["request"]["url"])
        self.assertEqual(redacted["request"]["items"][0]["cookie"], "[REDACTED]")
        self.assertEqual(redacted["request"]["items"][1]["safe"], "value")

    def test_redact_sensitive_text_redacts_json_like_strings(self):
        sample = '{"access_token":"abc123","headers":{"Authorization":"Bearer secret-secret-secret"}}'
        redacted = ru.redact_sensitive_text(sample)
        self.assertIn('"access_token": "[REDACTED]"', redacted)
        self.assertIn('"Authorization": "[REDACTED]"', redacted)
        self.assertNotIn('abc123', redacted)

    def test_summarize_tool_outcome_prefers_artifact_and_redacts(self):
        raw = {
            "status": "ok",
            "tool": "workflow_execute",
            "artifact": {
                "kind": "workflow_execution",
                "developer_summary": "Created workflow using https://example.com?token=abc123",
                "next_actions": [
                    "Reuse token=abc123 output",
                    "Run tests next",
                ],
            },
        }

        summary = ru.summarize_tool_outcome("workflow_execute", raw)
        self.assertEqual(summary["tool_name"], "workflow_execute")
        self.assertEqual(summary["status"], "ok")
        self.assertEqual(summary["artifact_kind"], "workflow_execution")
        self.assertIn("token=[REDACTED]", summary["summary"])
        self.assertIn("token=[REDACTED]", summary["next_actions"][0])
        self.assertNotIn("abc123", json.dumps(summary))

    def test_summarize_tool_outcome_keeps_fs_result_count(self):
        summary = ru.summarize_tool_outcome(
            "fs_search",
            {
                "status": "ok",
                "tool": "fs_search",
                "count": 3,
                "matches": [
                    {"path": "README.md", "line": 10, "text": "agent"},
                    {"path": "README.md", "line": 12, "text": "agent"},
                    {"path": "README.md", "line": 15, "text": "agent"},
                ],
            },
        )
        self.assertEqual(summary["tool_name"], "fs_search")
        self.assertEqual(summary["count"], 3)

    def test_summarize_tool_outcome_keeps_edit_activity_metadata(self):
        summary = ru.summarize_tool_outcome(
            "fs_patch",
            {
                "status": "ok",
                "tool": "fs_patch",
                "changed": True,
                "wrote": True,
                "dry_run": False,
                "edit_results": [
                    {"index": 1, "matches": 2, "regex": False, "count": 0},
                    {"index": 2, "matches": 1, "regex": True, "count": 1},
                ],
            },
        )
        self.assertEqual(summary["tool_name"], "fs_patch")
        self.assertEqual(summary["applied_matches"], 3)
        self.assertTrue(summary["changed"])
        self.assertTrue(summary["wrote"])
        self.assertFalse(summary["dry_run"])

    def test_build_execution_grounding_note_uses_verified_facts(self):
        note = ru.build_execution_grounding_note(
            "check repo progress",
            [
                {
                    "tool_name": "workflow_execute",
                    "status": "ok",
                    "verified": True,
                    "summary": "Workflow completed successfully.",
                    "next_actions": ["Run tests next"],
                },
                {
                    "tool_name": "run_command",
                    "status": "failed",
                    "verified": False,
                    "summary": "Command failed: token=abc123",
                    "next_actions": [],
                },
            ],
        )
        self.assertIn("Answer only from the verified execution facts below.", note)
        self.assertIn("workflow_execute | status=ok | verified=yes", note)
        self.assertIn("run_command | status=failed | verified=no", note)
        self.assertIn("token=[REDACTED]", note)
        self.assertNotIn("abc123", note)

    def test_build_turn_record_keeps_next_steps(self):
        record = ru.build_turn_record(
            "update workflow",
            [
                {
                    "tool_name": "workflow_execute",
                    "status": "ok",
                    "verified": True,
                    "summary": "Done.",
                    "next_actions": ["Run tests", "Write docs"],
                }
            ],
            "Updated workflow and next steps are run tests then write docs.",
        )
        self.assertEqual(record["completed_tools"], 1)
        self.assertEqual(record["failed_tools"], 0)
        self.assertEqual(record["next_steps"], ["Run tests", "Write docs"])
        self.assertIn("Updated workflow", record["developer_summary"])

    def test_build_response_override_normalizes_local_access_reason(self):
        override = ru.build_response_override(
            "local-access-fallback",
            "The drafted answer claimed local workspace access was unavailable, so the runtime replaced it with a direct workspace inspection summary.",
            source="local-access-fallback",
        )
        self.assertEqual(override["source"], "local-access-fallback")
        self.assertEqual(override["kind"], "local-access-fallback")
        self.assertIn("runtime replaced it with a direct workspace inspection summary", override["reason"])

    def test_build_turn_record_keeps_grounding_override_reason(self):
        record = ru.build_turn_record(
            "inspect repo",
            [
                {
                    "tool_name": "fs_list",
                    "status": "ok",
                    "verified": False,
                    "count": 3,
                    "summary": "count=3; path=agent",
                    "next_actions": ["Read files"],
                }
            ],
            "No files were found.",
            grounding_mismatch={
                "source": "grounding-mismatch",
                "kind": "file-list-results",
                "reason": "The drafted answer says no files were found, but fs_list returned one or more workspace entries.",
            },
        )
        self.assertEqual(record["grounding_override"]["source"], "grounding-mismatch")
        self.assertEqual(record["grounding_override"]["kind"], "file-list-results")
        self.assertIn("fs_list returned one or more workspace entries", record["grounding_override"]["reason"])

    def test_format_session_resume_note_renders_local_access_override(self):
        note = ru.format_session_resume_note(
            [
                {
                    "user_prompt": "inspect C:\\repo",
                    "developer_summary": "I can access your local workspace. I inspected `C:\\repo`.",
                    "grounding_override": {
                        "source": "local-access-fallback",
                        "kind": "local-access-fallback",
                        "reason": "The drafted answer claimed local workspace access was unavailable, so the runtime replaced it with a direct workspace inspection summary.",
                    },
                    "next_steps": ["Inspect specific files"],
                }
            ]
        )
        self.assertIn("Override: source=local-access-fallback - The drafted answer claimed local workspace access was unavailable", note)
        self.assertIn("Next: Inspect specific files", note)

    def test_format_session_resume_note_renders_recent_turns(self):
        note = ru.format_session_resume_note(
            [
                {
                    "user_prompt": "fix browser flow",
                    "developer_summary": "Fixed browser flow and saved artifact.",
                    "next_steps": ["Run smoke test", "Update docs"],
                },
                {
                    "user_prompt": "harden API outputs",
                    "developer_summary": "Redacted token=abc123 from API outputs.",
                    "grounding_override": {
                        "source": "grounding-mismatch",
                        "kind": "verification",
                        "reason": "The drafted answer said the result was not verified, but verified tool output exists for this turn.",
                    },
                    "next_steps": ["Verify unit tests"],
                },
            ]
        )
        self.assertIn("[Session resume state]", note)
        self.assertIn("Goal: fix browser flow", note)
        self.assertIn("Next: Run smoke test; Update docs", note)
        self.assertIn("Override: source=grounding-mismatch kind=verification - The drafted answer said the result was not verified", note)
        self.assertIn("token=[REDACTED]", note)
        self.assertNotIn("abc123", note)

    def test_collect_recent_next_steps_keeps_unique_order(self):
        actions = ru.collect_recent_next_steps(
            [
                {"next_steps": ["Run docs", "Ship patch"]},
                {"next_steps": ["Ship patch", "Verify tests"]},
            ],
            max_items=4,
        )
        self.assertEqual(actions, ["Ship patch", "Verify tests", "Run docs"])

    def test_format_last_turn_report_renders_summary_and_next_steps(self):
        report = ru.format_last_turn_report(
            [
                {
                    "user_prompt": "fix runtime",
                    "developer_summary": "Updated runtime grounding and next steps.",
                    "completed_tools": 2,
                    "failed_tools": 0,
                    "grounding_override": {
                        "source": "grounding-mismatch",
                        "kind": "workspace-change",
                        "reason": "The drafted answer says no workspace changes were made, but a workspace mutation tool completed successfully.",
                    },
                    "tool_summaries": [
                        {"tool_name": "workflow_execute", "status": "ok", "summary": "Workflow completed."}
                    ],
                    "next_steps": ["Run unit tests", "Update docs"],
                }
            ]
        )
        self.assertIn("[Last completed turn]", report)
        self.assertIn("Goal: fix runtime", report)
        self.assertIn("Done: Updated runtime grounding", report)
        self.assertIn("Grounding override: source=grounding-mismatch kind=workspace-change | The drafted answer says no workspace changes were made", report)
        self.assertIn("- workflow_execute | status=ok | Workflow completed.", report)
        self.assertIn("Next steps:", report)

    def test_augment_final_answer_with_grounding_appends_verified_block(self):
        answer = ru.augment_final_answer_with_grounding(
            "I updated the workflow runtime.",
            [
                {
                    "tool_name": "workflow_execute",
                    "status": "ok",
                    "summary": "Workflow completed using token=abc123.",
                    "next_actions": ["Run unit tests"],
                }
            ],
        )
        self.assertIn("I updated the workflow runtime.", answer)
        self.assertIn("Verified execution:", answer)
        self.assertIn("workflow_execute | status=ok", answer)
        self.assertIn("token=[REDACTED]", answer)
        self.assertIn("Next steps:", answer)
        self.assertNotIn("abc123", answer)

    def test_detect_final_answer_grounding_mismatch_for_failed_tests(self):
        mismatch = ru.detect_final_answer_grounding_mismatch(
            "All tests passed and everything is green.",
            [
                {
                    "tool_name": "run_command",
                    "status": "failed",
                    "verified": True,
                    "summary": "pytest failed with 2 failing tests.",
                    "next_actions": ["Fix failing tests"],
                }
            ],
        )
        self.assertIsNotNone(mismatch)
        self.assertEqual(mismatch["kind"], "tests")

    def test_detect_final_answer_grounding_mismatch_for_workspace_change_claim(self):
        mismatch = ru.detect_final_answer_grounding_mismatch(
            "No changes were made to the workspace.",
            [
                {
                    "tool_name": "fs_write",
                    "status": "ok",
                    "verified": True,
                    "summary": "Updated README.md.",
                    "next_actions": ["Run docs check"],
                }
            ],
        )
        self.assertIsNotNone(mismatch)
        self.assertEqual(mismatch["kind"], "workspace-change")

    def test_detect_final_answer_grounding_mismatch_for_positive_fs_list_results(self):
        mismatch = ru.detect_final_answer_grounding_mismatch(
            "No files were found in that folder.",
            [
                {
                    "tool_name": "fs_list",
                    "status": "ok",
                    "verified": False,
                    "count": 4,
                    "summary": "count=4; path=src",
                    "next_actions": ["Inspect matching files"],
                }
            ],
        )
        self.assertIsNotNone(mismatch)
        self.assertEqual(mismatch["kind"], "file-list-results")

    def test_detect_final_answer_grounding_mismatch_for_positive_fs_search_results(self):
        mismatch = ru.detect_final_answer_grounding_mismatch(
            "No results were found for that search.",
            [
                {
                    "tool_name": "fs_search",
                    "status": "ok",
                    "verified": False,
                    "count": 2,
                    "summary": "count=2; path=src/app.py",
                    "next_actions": ["Open matching lines"],
                }
            ],
        )
        self.assertIsNotNone(mismatch)
        self.assertEqual(mismatch["kind"], "file-search-results")

    def test_detect_final_answer_grounding_mismatch_for_successful_edit_activity(self):
        mismatch = ru.detect_final_answer_grounding_mismatch(
            "No edits were applied to the file.",
            [
                {
                    "tool_name": "fs_edit_lines",
                    "status": "ok",
                    "verified": False,
                    "changed": True,
                    "wrote": True,
                    "dry_run": False,
                    "summary": "Updated config lines.",
                    "next_actions": ["Run tests"],
                }
            ],
        )
        self.assertIsNotNone(mismatch)
        self.assertEqual(mismatch["kind"], "edit-activity")

    def test_detect_final_answer_grounding_mismatch_for_successful_patch_matches(self):
        mismatch = ru.detect_final_answer_grounding_mismatch(
            "No matches were found for the patch.",
            [
                {
                    "tool_name": "fs_patch",
                    "status": "ok",
                    "verified": False,
                    "changed": True,
                    "wrote": True,
                    "dry_run": False,
                    "applied_matches": 2,
                    "summary": "Applied patch to README.md.",
                    "next_actions": ["Verify docs"],
                }
            ],
        )
        self.assertIsNotNone(mismatch)
        self.assertEqual(mismatch["kind"], "edit-matches")

    def test_detect_final_answer_grounding_mismatch_does_not_fire_for_dry_run_edit(self):
        mismatch = ru.detect_final_answer_grounding_mismatch(
            "No edits were applied to the file.",
            [
                {
                    "tool_name": "fs_edit_lines",
                    "status": "ok",
                    "verified": False,
                    "changed": True,
                    "wrote": False,
                    "dry_run": True,
                    "summary": "Previewed config line edits.",
                    "next_actions": ["Apply the change if approved"],
                }
            ],
        )
        self.assertIsNone(mismatch)

    def test_reconcile_final_answer_with_grounding_replaces_obvious_mismatch(self):
        answer = ru.reconcile_final_answer_with_grounding(
            "All tests passed and no changes were made.",
            [
                {
                    "tool_name": "fs_write",
                    "status": "ok",
                    "verified": True,
                    "summary": "Updated README.md.",
                    "next_actions": ["Run tests"],
                },
                {
                    "tool_name": "run_command",
                    "status": "failed",
                    "verified": True,
                    "summary": "pytest failed with 1 failing test.",
                    "next_actions": ["Fix failing test"],
                },
            ],
        )
        self.assertIn("grounded directly in tool output", answer)
        self.assertIn("Verified execution:", answer)
        self.assertIn("fs_write | status=ok", answer)
        self.assertIn("run_command | status=failed", answer)
        self.assertNotIn("All tests passed", answer)
        self.assertNotIn("no changes were made", answer.lower())

    def test_reconcile_final_answer_with_grounding_replaces_false_no_files_claim(self):
        answer = ru.reconcile_final_answer_with_grounding(
            "No files were found in the workspace.",
            [
                {
                    "tool_name": "fs_list",
                    "status": "ok",
                    "verified": False,
                    "count": 3,
                    "summary": "count=3; path=agent",
                    "next_actions": ["Read matching files"],
                }
            ],
        )
        self.assertIn("grounded directly in tool output", answer)
        self.assertIn("fs_list | status=ok", answer)
        self.assertNotIn("No files were found", answer)

    def test_reconcile_final_answer_with_grounding_replaces_false_no_edits_claim(self):
        answer = ru.reconcile_final_answer_with_grounding(
            "No edits were applied to the file.",
            [
                {
                    "tool_name": "fs_insert_lines",
                    "status": "ok",
                    "verified": False,
                    "changed": True,
                    "wrote": True,
                    "dry_run": False,
                    "summary": "Inserted 2 lines into settings.py.",
                    "next_actions": ["Run tests"],
                }
            ],
        )
        self.assertIn("grounded directly in tool output", answer)
        self.assertIn("fs_insert_lines | status=ok", answer)
        self.assertNotIn("No edits were applied", answer)

    def test_reconcile_final_answer_with_grounding_metadata_returns_override_details(self):
        result = ru.reconcile_final_answer_with_grounding_metadata(
            "No files were found in the workspace.",
            [
                {
                    "tool_name": "fs_list",
                    "status": "ok",
                    "verified": False,
                    "count": 2,
                    "summary": "count=2; path=agent",
                    "next_actions": ["Read matching files"],
                }
            ],
        )
        self.assertIn("answer", result)
        self.assertIn("grounding_mismatch", result)
        self.assertEqual(result["grounding_mismatch"]["source"], "grounding-mismatch")
        self.assertEqual(result["grounding_mismatch"]["kind"], "file-list-results")
        self.assertIn("fs_list returned one or more workspace entries", result["grounding_mismatch"]["reason"])

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
