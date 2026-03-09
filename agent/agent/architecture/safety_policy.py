import json
import os
from pathlib import Path

try:
    from architecture.safety_confirm import issue_confirm_token, validate_confirm_token
    from architecture.safety_registry import get_tool_safety_rule
    from architecture.safety_types import ActionClass, PolicyDecision, SafetyEvaluation
except Exception:
    from .safety_confirm import issue_confirm_token, validate_confirm_token
    from .safety_registry import get_tool_safety_rule
    from .safety_types import ActionClass, PolicyDecision, SafetyEvaluation


ALLOW_DESTRUCTIVE_TOOLS_ENV = "AGENT_ALLOW_DESTRUCTIVE_TOOLS"
ALLOW_BROWSER_CODE_EXECUTION_ENV = "AGENT_ALLOW_BROWSER_CODE_EXECUTION"
SENSITIVE_PATH_MARKERS = (
    ".env",
    ".agent-state",
    "memory",
    ".github/workflows",
    ".github/codeowners",
    "security.md",
    "release_checklist.md",
)
DANGEROUS_BROWSER_HINTS = (
    "delete",
    "remove",
    "purchase",
    "checkout",
    "pay",
    "submit",
    "send",
    "confirm order",
    "place order",
    "merge",
    "publish",
)
AUTH_ARGUMENT_KEYS = {"oauth_profile", "bearer_token", "auth", "client_secret", "refresh_token"}
FILE_PATH_ARGUMENT_KEYS = {"path", "source", "destination", "cwd", "file_path", "save_path"}


def _to_bool(value, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _normalized_args(arguments):
    return arguments if isinstance(arguments, dict) else {}


def _collect_path_values(arguments: dict) -> list[str]:
    values = []
    for key, value in arguments.items():
        if key in FILE_PATH_ARGUMENT_KEYS and isinstance(value, str) and value.strip():
            values.append(value.strip())
    return values


def _is_sensitive_path(path_value: str) -> bool:
    normalized = str(path_value or "").replace("\\", "/").strip().lower()
    if not normalized:
        return False
    return any(marker in normalized for marker in SENSITIVE_PATH_MARKERS)


def _path_exists(workspace_root: Path, path_value: str) -> bool:
    try:
        candidate = Path(path_value)
        if not candidate.is_absolute():
            candidate = (workspace_root / candidate).resolve()
        else:
            candidate = candidate.resolve()
        return candidate.exists()
    except Exception:
        return False


def _serialized_argument_text(arguments: dict) -> str:
    try:
        return json.dumps(arguments, ensure_ascii=True, sort_keys=True).lower()
    except Exception:
        return str(arguments).lower()


def _build_preview_summary(tool_name: str, arguments: dict, workspace_root: Path) -> dict:
    summary = {"tool_name": tool_name}
    if tool_name == "workflow_execute":
        steps = arguments.get("steps", [])
        summary["steps"] = len(steps) if isinstance(steps, list) else 0
        return summary

    path_values = _collect_path_values(arguments)
    if path_values:
        summary["paths"] = path_values[:4]
        summary["sensitive_paths"] = [value for value in path_values if _is_sensitive_path(value)]
        summary["existing_paths"] = [value for value in path_values if _path_exists(workspace_root, value)]

    if tool_name == "run_command":
        summary["command"] = str(arguments.get("command", ""))[:240]
        summary["security_mode"] = str(arguments.get("security_mode", "restricted"))
    elif tool_name == "fs_write":
        summary["append"] = bool(arguments.get("append", False))
        summary["content_chars"] = len(str(arguments.get("content", "")))
    elif tool_name == "fs_patch":
        summary["edit_count"] = len(arguments.get("edits", [])) if isinstance(arguments.get("edits"), list) else 0
    elif tool_name.startswith("browser_"):
        summary["argument_text"] = _serialized_argument_text(arguments)[:240]
    return summary


def _allow_with_optional_confirmation(
    tool_name: str,
    arguments: dict,
    action_class: str,
    risk_level: str,
    reason_codes: list[str],
    requires_verification: bool,
    preview_summary: dict | None,
    base_decision: str,
) -> SafetyEvaluation:
    confirm_requested = _to_bool(arguments.get("confirm", False), False)
    confirm_token = str(arguments.get("confirm_token", "")).strip()
    if base_decision in {PolicyDecision.PREVIEW_REQUIRED.value, PolicyDecision.CONFIRM_REQUIRED.value}:
        if confirm_requested and validate_confirm_token(tool_name, arguments, confirm_token):
            return SafetyEvaluation(
                tool_name=tool_name,
                action_class=action_class,
                risk_level=risk_level,
                decision=PolicyDecision.ALLOW_WITH_VERIFICATION.value,
                reason_codes=reason_codes + ["confirmed"],
                requires_verification=True,
            )
        return SafetyEvaluation(
            tool_name=tool_name,
            action_class=action_class,
            risk_level=risk_level,
            decision=base_decision,
            reason_codes=reason_codes,
            requires_verification=requires_verification,
            preview_summary=preview_summary,
            confirm_token=issue_confirm_token(tool_name, arguments),
        )
    return SafetyEvaluation(
        tool_name=tool_name,
        action_class=action_class,
        risk_level=risk_level,
        decision=base_decision,
        reason_codes=reason_codes,
        requires_verification=requires_verification,
        preview_summary=preview_summary,
    )


def _evaluate_workflow(tool_name: str, arguments: dict, workspace_root: Path) -> SafetyEvaluation:
    steps = arguments.get("steps", [])
    if not isinstance(steps, list) or not steps:
        return SafetyEvaluation(
            tool_name=tool_name,
            action_class=ActionClass.BROAD_LOCAL_WRITE.value,
            risk_level="medium",
            decision=PolicyDecision.ALLOW_WITH_VERIFICATION.value,
            reason_codes=["empty-workflow"],
            requires_verification=True,
        )

    decisions = []
    for step in steps[:50]:
        if not isinstance(step, dict):
            continue
        nested_tool = str(step.get("tool_name", "")).strip()
        nested_args = step.get("arguments", {}) or {}
        nested = evaluate_tool_call(nested_tool, nested_args, workspace_root)
        decisions.append(
            {
                "tool_name": nested_tool,
                "decision": nested.decision,
                "action_class": nested.action_class,
            }
        )
        if nested.decision == PolicyDecision.BLOCKED.value:
            return _allow_with_optional_confirmation(
                tool_name=tool_name,
                arguments=arguments,
                action_class=ActionClass.BROAD_LOCAL_WRITE.value,
                risk_level="high",
                reason_codes=["workflow-contains-blocked-step"],
                requires_verification=True,
                preview_summary={"steps": decisions},
                base_decision=PolicyDecision.CONFIRM_REQUIRED.value,
            )
        if nested.decision in {PolicyDecision.CONFIRM_REQUIRED.value, PolicyDecision.PREVIEW_REQUIRED.value}:
            return _allow_with_optional_confirmation(
                tool_name=tool_name,
                arguments=arguments,
                action_class=ActionClass.BROAD_LOCAL_WRITE.value,
                risk_level="high",
                reason_codes=["workflow-contains-state-changing-steps"],
                requires_verification=True,
                preview_summary={"steps": decisions},
                base_decision=PolicyDecision.PREVIEW_REQUIRED.value,
            )

    return SafetyEvaluation(
        tool_name=tool_name,
        action_class=ActionClass.BROAD_LOCAL_WRITE.value,
        risk_level="medium",
        decision=PolicyDecision.ALLOW_WITH_VERIFICATION.value,
        reason_codes=["workflow-read-only-or-verified"],
        requires_verification=True,
        preview_summary={"steps": decisions},
    )


def evaluate_tool_call(tool_name: str, arguments: dict, workspace_root: Path) -> SafetyEvaluation:
    clean_name = str(tool_name or "").strip()
    clean_args = _normalized_args(arguments)

    if clean_name == "call_tool":
        nested_name = str(clean_args.get("tool_name", "")).strip()
        nested_args = clean_args.get("arguments", {}) or {}
        if not nested_name:
            return SafetyEvaluation(
                tool_name=clean_name,
                action_class=ActionClass.BROAD_LOCAL_WRITE.value,
                risk_level="medium",
                decision=PolicyDecision.BLOCKED.value,
                reason_codes=["missing-nested-tool-name"],
            )
        nested = evaluate_tool_call(nested_name, nested_args, workspace_root)
        nested.reason_codes.insert(0, "indirect-tool-call")
        return nested

    if clean_name == "workflow_execute":
        return _evaluate_workflow(clean_name, clean_args, workspace_root)

    rule = get_tool_safety_rule(clean_name)
    preview_summary = _build_preview_summary(clean_name, clean_args, workspace_root)
    serialized_args = _serialized_argument_text(clean_args)
    path_values = _collect_path_values(clean_args)
    sensitive_paths = [value for value in path_values if _is_sensitive_path(value)]

    if rule.action_class == ActionClass.READ_ONLY.value:
        if clean_name == "web_fetch" and any(key in clean_args for key in AUTH_ARGUMENT_KEYS):
            return _allow_with_optional_confirmation(
                tool_name=clean_name,
                arguments=clean_args,
                action_class=ActionClass.EXTERNAL_SIDE_EFFECT.value,
                risk_level="high",
                reason_codes=["authenticated-network-read"],
                requires_verification=True,
                preview_summary=preview_summary,
                base_decision=PolicyDecision.CONFIRM_REQUIRED.value,
            )
        return SafetyEvaluation(
            tool_name=clean_name,
            action_class=rule.action_class,
            risk_level="low",
            decision=PolicyDecision.ALLOW.value,
            reason_codes=["read-only"],
            requires_verification=False,
        )

    if clean_name in {"fs_edit_lines", "fs_insert_lines", "fs_patch"}:
        if _to_bool(clean_args.get("dry_run", False), False):
            return SafetyEvaluation(
                tool_name=clean_name,
                action_class=ActionClass.READ_ONLY.value,
                risk_level="low",
                decision=PolicyDecision.ALLOW.value,
                reason_codes=["dry-run"],
            )
        if sensitive_paths:
            return _allow_with_optional_confirmation(
                tool_name=clean_name,
                arguments=clean_args,
                action_class=ActionClass.BROAD_LOCAL_WRITE.value,
                risk_level="high",
                reason_codes=["sensitive-path-write"],
                requires_verification=True,
                preview_summary=preview_summary,
                base_decision=PolicyDecision.PREVIEW_REQUIRED.value,
            )
        return SafetyEvaluation(
            tool_name=clean_name,
            action_class=ActionClass.SCOPED_REVERSIBLE_WRITE.value,
            risk_level="medium",
            decision=PolicyDecision.ALLOW_WITH_VERIFICATION.value,
            reason_codes=["scoped-diffable-write"],
            requires_verification=True,
        )

    if clean_name == "fs_write":
        if sensitive_paths or any(_path_exists(workspace_root, value) for value in path_values):
            return _allow_with_optional_confirmation(
                tool_name=clean_name,
                arguments=clean_args,
                action_class=ActionClass.BROAD_LOCAL_WRITE.value,
                risk_level="high",
                reason_codes=["overwrite-or-sensitive-write"],
                requires_verification=True,
                preview_summary=preview_summary,
                base_decision=PolicyDecision.CONFIRM_REQUIRED.value,
            )
        return _allow_with_optional_confirmation(
            tool_name=clean_name,
            arguments=clean_args,
            action_class=ActionClass.BROAD_LOCAL_WRITE.value,
            risk_level="medium",
            reason_codes=["new-file-write"],
            requires_verification=True,
            preview_summary=preview_summary,
            base_decision=PolicyDecision.PREVIEW_REQUIRED.value,
        )

    if clean_name in {"fs_copy", "fs_move"}:
        overwrite = _to_bool(clean_args.get("overwrite", False), False)
        base_decision = PolicyDecision.CONFIRM_REQUIRED.value if (overwrite or sensitive_paths) else PolicyDecision.PREVIEW_REQUIRED.value
        reasons = ["sensitive-or-overwrite-copy"] if base_decision == PolicyDecision.CONFIRM_REQUIRED.value else ["copy-or-move-preview"]
        risk = "high" if base_decision == PolicyDecision.CONFIRM_REQUIRED.value else "medium"
        return _allow_with_optional_confirmation(
            tool_name=clean_name,
            arguments=clean_args,
            action_class=ActionClass.BROAD_LOCAL_WRITE.value,
            risk_level=risk,
            reason_codes=reasons,
            requires_verification=True,
            preview_summary=preview_summary,
            base_decision=base_decision,
        )

    if clean_name == "fs_delete":
        if not _to_bool(os.getenv(ALLOW_DESTRUCTIVE_TOOLS_ENV, "0"), False):
            return SafetyEvaluation(
                tool_name=clean_name,
                action_class=ActionClass.DESTRUCTIVE.value,
                risk_level="critical",
                decision=PolicyDecision.BLOCKED.value,
                reason_codes=["destructive-tools-disabled"],
                preview_summary=preview_summary,
            )
        return _allow_with_optional_confirmation(
            tool_name=clean_name,
            arguments=clean_args,
            action_class=ActionClass.DESTRUCTIVE.value,
            risk_level="critical",
            reason_codes=["destructive-delete"],
            requires_verification=True,
            preview_summary=preview_summary,
            base_decision=PolicyDecision.CONFIRM_REQUIRED.value,
        )

    if clean_name == "run_command":
        security_mode = str(clean_args.get("security_mode", "restricted")).strip().lower() or "restricted"
        allow_dangerous = _to_bool(clean_args.get("allow_dangerous", False), False)
        if allow_dangerous:
            return _allow_with_optional_confirmation(
                tool_name=clean_name,
                arguments=clean_args,
                action_class=ActionClass.DESTRUCTIVE.value,
                risk_level="critical",
                reason_codes=["dangerous-command-requested"],
                requires_verification=True,
                preview_summary=preview_summary,
                base_decision=PolicyDecision.CONFIRM_REQUIRED.value,
            )
        if security_mode == "permissive":
            return _allow_with_optional_confirmation(
                tool_name=clean_name,
                arguments=clean_args,
                action_class=ActionClass.BROAD_LOCAL_WRITE.value,
                risk_level="high",
                reason_codes=["permissive-command-mode"],
                requires_verification=True,
                preview_summary=preview_summary,
                base_decision=PolicyDecision.PREVIEW_REQUIRED.value,
            )
        return SafetyEvaluation(
            tool_name=clean_name,
            action_class=ActionClass.SCOPED_REVERSIBLE_WRITE.value,
            risk_level="medium",
            decision=PolicyDecision.ALLOW_WITH_VERIFICATION.value,
            reason_codes=["restricted-command-mode"],
            requires_verification=True,
        )

    if clean_name in {"oauth_set_profile", "oauth_get_token", "memory_log", "memory_promote"}:
        risk = "high" if clean_name != "oauth_get_token" else "critical"
        return _allow_with_optional_confirmation(
            tool_name=clean_name,
            arguments=clean_args,
            action_class=ActionClass.EXTERNAL_SIDE_EFFECT.value if clean_name == "oauth_get_token" else ActionClass.BROAD_LOCAL_WRITE.value,
            risk_level=risk,
            reason_codes=["sensitive-persistent-or-auth-action"],
            requires_verification=True,
            preview_summary=preview_summary,
            base_decision=PolicyDecision.CONFIRM_REQUIRED.value,
        )

    if clean_name in {"memory_reindex", "browser_take_screenshot", "browser_pdf_save", "browser_close_blank_tabs"}:
        return SafetyEvaluation(
            tool_name=clean_name,
            action_class=rule.action_class,
            risk_level="medium",
            decision=PolicyDecision.ALLOW_WITH_VERIFICATION.value,
            reason_codes=["bounded-local-artifact-or-maintenance"],
            requires_verification=True,
        )

    if clean_name in {"browser_evaluate", "browser_run_code"}:
        if not _to_bool(os.getenv(ALLOW_BROWSER_CODE_EXECUTION_ENV, "0"), False):
            return SafetyEvaluation(
                tool_name=clean_name,
                action_class=ActionClass.DESTRUCTIVE.value,
                risk_level="critical",
                decision=PolicyDecision.BLOCKED.value,
                reason_codes=["browser-code-execution-disabled"],
                preview_summary=preview_summary,
            )
        return _allow_with_optional_confirmation(
            tool_name=clean_name,
            arguments=clean_args,
            action_class=ActionClass.DESTRUCTIVE.value,
            risk_level="critical",
            reason_codes=["browser-code-execution"],
            requires_verification=True,
            preview_summary=preview_summary,
            base_decision=PolicyDecision.CONFIRM_REQUIRED.value,
        )

    if clean_name == "browser_file_upload":
        return _allow_with_optional_confirmation(
            tool_name=clean_name,
            arguments=clean_args,
            action_class=ActionClass.BROAD_LOCAL_WRITE.value,
            risk_level="high",
            reason_codes=["browser-file-upload"],
            requires_verification=True,
            preview_summary=preview_summary,
            base_decision=PolicyDecision.CONFIRM_REQUIRED.value,
        )

    if clean_name.startswith("browser_"):
        if any(hint in serialized_args for hint in DANGEROUS_BROWSER_HINTS):
            return _allow_with_optional_confirmation(
                tool_name=clean_name,
                arguments=clean_args,
                action_class=ActionClass.EXTERNAL_SIDE_EFFECT.value,
                risk_level="high",
                reason_codes=["browser-dangerous-intent-hint"],
                requires_verification=True,
                preview_summary=preview_summary,
                base_decision=PolicyDecision.CONFIRM_REQUIRED.value,
            )
        return SafetyEvaluation(
            tool_name=clean_name,
            action_class=rule.action_class,
            risk_level="medium",
            decision=PolicyDecision.ALLOW_WITH_VERIFICATION.value,
            reason_codes=["browser-bounded-action"],
            requires_verification=True,
        )

    return SafetyEvaluation(
        tool_name=clean_name,
        action_class=rule.action_class,
        risk_level="medium",
        decision=PolicyDecision.ALLOW_WITH_VERIFICATION.value,
        reason_codes=["default-guarded-allow"],
        requires_verification=True,
    )
