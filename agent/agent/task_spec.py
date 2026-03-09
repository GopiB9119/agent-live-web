import json
import re
from typing import Any


BROWSER_KEYWORDS = {
    "browser",
    "website",
    "web site",
    "page",
    "dom",
    "playwright",
    "selector",
    "navigate",
    "click",
    "type",
    "form",
    "modal",
    "dialog",
    "screenshot",
}

WORKSPACE_KEYWORDS = {
    "repo",
    "repository",
    "workspace",
    "codebase",
    "file",
    "folder",
    "directory",
    "project",
    "module",
    "function",
    "class",
    "test",
    "refactor",
    "lint",
    "typecheck",
    "build",
}

PYTHON_KEYWORDS = {
    "python",
    "pylance",
    "pytest",
    "venv",
    "pip",
    "import",
    "requirements",
}

GITHUB_KEYWORDS = {
    "github",
    "pull request",
    "pr ",
    "issue",
    "branch",
    "commit",
    "review",
    "merge",
}

DEBUG_KEYWORDS = {
    "debug",
    "fix",
    "failing",
    "failure",
    "error",
    "trace",
    "stack",
    "broken",
    "triage",
    "investigate",
}

TEST_KEYWORDS = {
    "test",
    "verify",
    "validation",
    "validate",
    "assert",
    "check",
}

READONLY_KEYWORDS = {
    "inspect",
    "read",
    "list",
    "show",
    "analyze",
    "understand",
    "scan",
    "find",
    "search",
}

MUTATE_KEYWORDS = {
    "edit",
    "update",
    "change",
    "modify",
    "rewrite",
    "create",
    "generate",
    "implement",
    "add",
    "build",
}

ACTION_KEYWORDS = {
    "navigate": ["navigate", "open page", "go to", "visit"],
    "search": ["search", "find", "lookup", "look up"],
    "read": ["read", "inspect", "understand", "analyze", "scan", "list", "show"],
    "edit": ["edit", "update", "change", "modify", "rewrite", "refactor"],
    "interact": ["click", "type", "fill", "select", "press", "hover"],
    "create": ["create", "generate", "write", "add", "build"],
    "test": ["test", "verify", "validate", "check", "assert"],
    "debug": ["debug", "fix", "repair", "troubleshoot", "investigate"],
    "submit": ["submit", "send", "checkout", "purchase", "merge", "push"],
    "delete": ["delete", "remove", "clear"],
    "screenshot": ["screenshot", "capture"],
    "trace": ["trace", "network", "console"],
    "upload": ["upload", "attach file"],
    "download": ["download", "save pdf", "export pdf", "pdf"],
    "fetch": ["fetch", "request", "api", "http"],
}


def _as_text(value: Any) -> str:
    return str(value or "").strip()


def _normalize_string(value: Any, default: str = "unknown") -> str:
    text = _as_text(value)
    return text if text else default


def _normalize_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


def _coerce_dict(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def _coerce_list(value: Any) -> list:
    return value if isinstance(value, list) else []


def _contains_keyword(text: str, keywords: set[str]) -> bool:
    lowered = text.lower()
    return any(keyword in lowered for keyword in keywords)


def _extract_json_payload(text: str) -> dict | None:
    if not isinstance(text, str):
        return None
    stripped = text.strip()
    if not stripped:
        return None

    candidates = [stripped]
    fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, flags=re.IGNORECASE | re.DOTALL)
    candidates.extend(candidate.strip() for candidate in fenced if candidate.strip())

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except Exception:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _normalize_environment(text: str, explicit: str = "") -> str:
    value = _normalize_string(explicit, default="").lower()
    if value:
        if value in {"prod", "production", "live"}:
            return "production"
        if value in {"staging", "stage"}:
            return "staging"
        if value in {"dev", "development"}:
            return "development"
        return value

    lowered = text.lower()
    if any(token in lowered for token in {"production", "prod", "live site", "real site"}):
        return "production"
    if "staging" in lowered or "stage" in lowered:
        return "staging"
    if "local" in lowered or "localhost" in lowered:
        return "local"
    if "dev" in lowered or "development" in lowered:
        return "development"
    return "unknown"


def _extract_requested_actions(text: str, operations: list[dict]) -> list[str]:
    lowered = text.lower()
    actions = []
    for action, keywords in ACTION_KEYWORDS.items():
        if any(keyword in lowered for keyword in keywords):
            actions.append(action)
    for item in operations:
        if not isinstance(item, dict):
            continue
        action = _normalize_string(item.get("action"), default="").lower()
        if action and action not in actions:
            actions.append(action)
    return actions


def _infer_task_mode(text: str, requested_actions: list[str], no_submit: bool, explicit: str = "") -> str:
    mode = _normalize_string(explicit, default="").lower()
    if mode in {"inspect", "draft", "test", "mutate", "submit", "debug"}:
        return mode

    lowered = text.lower()
    if "debug" in lowered or "debug" in requested_actions:
        return "debug"
    if not no_submit and ("submit" in lowered or "submit" in requested_actions):
        return "submit"
    if "test" in lowered or "test" in requested_actions:
        return "test"
    if "interact" in requested_actions or _contains_keyword(lowered, MUTATE_KEYWORDS):
        return "mutate"
    if _contains_keyword(lowered, READONLY_KEYWORDS):
        return "inspect"
    return "draft"


def _infer_primary_kind(
    *,
    wants_browser: bool,
    wants_workspace: bool,
    wants_python: bool,
    wants_github: bool,
    wants_diagnostics: bool,
) -> str:
    if wants_browser and wants_workspace:
        return "hybrid"
    if wants_browser:
        return "browser"
    if wants_github:
        return "platform"
    if wants_python:
        return "python"
    if wants_diagnostics:
        return "diagnostics"
    return "repo"


def _infer_risk_level(task_mode: str, environment: str, allow_delete: bool, allow_submit: bool, allow_write: bool) -> str:
    if environment == "production" or allow_delete or allow_submit or task_mode == "submit":
        return "high"
    if task_mode in {"mutate", "debug", "test"} or allow_write:
        return "medium"
    return "low"


def _default_task_spec(task_text: str) -> dict:
    return {
        "source": "text",
        "task": _as_text(task_text),
        "task_mode": "draft",
        "primary_kind": "repo",
        "developer_level": "standard",
        "environment": "unknown",
        "data_sensitivity": "unknown",
        "no_submit": False,
        "risk_level": "low",
        "requested_actions": [],
        "surfaces_hint": [],
        "workspace": {},
        "website": {},
        "operations": [],
        "validation": {},
        "safety": {},
        "limits": {},
        "reason_codes": [],
    }


def build_task_spec(task_input: Any) -> dict:
    if isinstance(task_input, dict):
        payload = task_input
        raw_text = _as_text(payload.get("task") or payload.get("goal") or payload.get("objective"))
    else:
        raw_text = _as_text(task_input)
        payload = _extract_json_payload(raw_text)

    if isinstance(payload, dict):
        workspace = _coerce_dict(payload.get("workspace"))
        website = _coerce_dict(payload.get("website"))
        validation = _coerce_dict(payload.get("validation"))
        safety = _coerce_dict(payload.get("safety"))
        limits = _coerce_dict(payload.get("limits"))
        operations = [item for item in _coerce_list(payload.get("operations")) if isinstance(item, dict)]
        task_text = _as_text(payload.get("task") or payload.get("goal") or payload.get("objective") or raw_text)
        requested_actions = _extract_requested_actions(task_text, operations)
        allow_write = _normalize_bool(safety.get("allow_write"), default=bool(workspace))
        allow_delete = _normalize_bool(safety.get("allow_delete"), default=False)
        allow_submit = _normalize_bool(safety.get("allow_submit"), default=False)
        no_submit = _normalize_bool(payload.get("no_submit"), default=False)
        no_submit = no_submit or (not allow_submit) or _normalize_bool(validation.get("must_not_submit_without_confirm"), default=False)
        environment = _normalize_environment(task_text, explicit=website.get("environment") or payload.get("environment"))
        wants_browser = bool(website) or any(action in {"navigate", "search", "submit", "upload", "download"} for action in requested_actions)
        wants_workspace = bool(workspace) or any(path_key in payload for path_key in {"workspace", "target_files", "focus_paths"})
        wants_python = _contains_keyword(task_text, PYTHON_KEYWORDS) or bool(payload.get("python"))
        wants_github = _contains_keyword(task_text, GITHUB_KEYWORDS) or bool(payload.get("github"))
        wants_diagnostics = bool(validation) or _contains_keyword(task_text, DEBUG_KEYWORDS | TEST_KEYWORDS)
        task_mode = _infer_task_mode(task_text, requested_actions, no_submit, explicit=payload.get("task_mode") or payload.get("mode"))
        primary_kind = _infer_primary_kind(
            wants_browser=wants_browser,
            wants_workspace=wants_workspace,
            wants_python=wants_python,
            wants_github=wants_github,
            wants_diagnostics=wants_diagnostics,
        )
        risk_level = _infer_risk_level(task_mode, environment, allow_delete, allow_submit, allow_write)
        reason_codes = []
        if wants_browser:
            reason_codes.append("browser_context_requested")
        if wants_workspace:
            reason_codes.append("workspace_context_requested")
        if wants_diagnostics:
            reason_codes.append("verification_or_debug_requested")

        return {
            "source": "structured",
            "task": task_text,
            "task_mode": task_mode,
            "primary_kind": primary_kind,
            "developer_level": _normalize_string(payload.get("developer_level"), default="standard"),
            "environment": environment,
            "data_sensitivity": _normalize_string(safety.get("data_sensitivity"), default="unknown"),
            "no_submit": no_submit,
            "risk_level": risk_level,
            "requested_actions": requested_actions,
            "surfaces_hint": [kind for kind, enabled in {
                "workspace": wants_workspace,
                "browser": wants_browser,
                "python": wants_python,
                "github": wants_github,
                "diagnostics": wants_diagnostics,
            }.items() if enabled],
            "workspace": workspace,
            "website": website,
            "operations": operations,
            "validation": validation,
            "safety": {
                "allow_write": allow_write,
                "allow_delete": allow_delete,
                "allow_submit": allow_submit,
                **safety,
            },
            "limits": limits,
            "reason_codes": reason_codes,
        }

    task_spec = _default_task_spec(raw_text)
    if not raw_text:
        task_spec["reason_codes"] = ["empty_request"]
        return task_spec

    requested_actions = _extract_requested_actions(raw_text, [])
    wants_browser = _contains_keyword(raw_text, BROWSER_KEYWORDS) or "http://" in raw_text.lower() or "https://" in raw_text.lower()
    wants_workspace = _contains_keyword(raw_text, WORKSPACE_KEYWORDS) or bool(re.search(r"[A-Za-z]:\\", raw_text))
    wants_python = _contains_keyword(raw_text, PYTHON_KEYWORDS)
    wants_github = _contains_keyword(raw_text, GITHUB_KEYWORDS)
    wants_diagnostics = _contains_keyword(raw_text, DEBUG_KEYWORDS | TEST_KEYWORDS)
    no_submit = any(token in raw_text.lower() for token in {"dry run", "dry-run", "without submit", "no submit", "do not submit", "before submit"})
    task_mode = _infer_task_mode(raw_text, requested_actions, no_submit)
    primary_kind = _infer_primary_kind(
        wants_browser=wants_browser,
        wants_workspace=wants_workspace,
        wants_python=wants_python,
        wants_github=wants_github,
        wants_diagnostics=wants_diagnostics,
    )
    environment = _normalize_environment(raw_text)
    risk_level = _infer_risk_level(task_mode, environment, False, not no_submit and task_mode == "submit", task_mode in {"draft", "mutate", "debug"})

    task_spec.update(
        {
            "task_mode": task_mode,
            "primary_kind": primary_kind,
            "environment": environment,
            "no_submit": no_submit,
            "risk_level": risk_level,
            "requested_actions": requested_actions,
            "surfaces_hint": [kind for kind, enabled in {
                "workspace": wants_workspace,
                "browser": wants_browser,
                "python": wants_python,
                "github": wants_github,
                "diagnostics": wants_diagnostics,
            }.items() if enabled],
            "reason_codes": [kind for kind, enabled in {
                "browser_context_requested": wants_browser,
                "workspace_context_requested": wants_workspace,
                "verification_or_debug_requested": wants_diagnostics,
                "python_context_requested": wants_python,
                "github_context_requested": wants_github,
            }.items() if enabled],
        }
    )
    return task_spec
