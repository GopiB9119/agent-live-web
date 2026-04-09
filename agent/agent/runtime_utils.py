import json
import os
import re
import socket
import ipaddress
from datetime import datetime, timezone
from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
NOISE_DIR_NAMES = {".git", "node_modules", "__pycache__", ".venv", "venv", ".mypy_cache", ".pytest_cache"}
BINARY_SUFFIXES = {
    ".pyc", ".pyd", ".so", ".dll", ".exe", ".bin", ".dat", ".db",
    ".zip", ".gz", ".7z", ".rar", ".tar",
    ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".ico",
    ".mp3", ".wav", ".ogg", ".mp4", ".mkv", ".avi", ".mov",
    ".pdf", ".woff", ".woff2", ".ttf", ".otf",
}
MEMORY_DIR = WORKSPACE_ROOT / "memory"
LONG_TERM_MEMORY_FILE = WORKSPACE_ROOT / "MEMORY.md"
MEMORY_VECTOR_INDEX_FILE = MEMORY_DIR / ".vector_index.json"
MEMORY_VECTOR_DIM = 192
RUN_COMMAND_SECURITY_MODE_DEFAULT = os.environ.get("AGENT_RUN_COMMAND_SECURITY_MODE", "restricted").strip().lower() or "restricted"
RUN_COMMAND_ALLOW_DANGEROUS_ENV = "AGENT_ALLOW_DANGEROUS_COMMANDS"
WEB_FETCH_ALLOW_PRIVATE_ENV = "AGENT_WEB_FETCH_ALLOW_PRIVATE_HOSTS"
MAX_MEMORY_LOG_CHARS = 4000
SENSITIVE_KEY_NAME_PATTERN = re.compile(
    r"(?i)(api[_-]?key|token|secret|password|passwd|pwd|cookie|authorization|session|bearer|credential|otp|pin|client_secret|access[_-]?token|refresh[_-]?token)"
)
SENSITIVE_QUERY_PARAM_PATTERN = re.compile(
    r"([?&](?:api[_-]?key|token|secret|password|passwd|pwd|cookie|authorization|auth|session|bearer|credential|otp|pin|client_secret|access[_-]?token|refresh[_-]?token|code)=)([^&#\s]+)",
    re.IGNORECASE,
)
URL_CREDENTIAL_PATTERN = re.compile(r"(?i)(https?://)([^/@\s]+(?::[^/@\s]+)?@)")
SENSITIVE_VALUE_PATTERNS = [
    re.compile(r"(?i)\b(bearer)\s+[A-Za-z0-9._\-]{12,}"),
    re.compile(r"(?i)\b([A-Za-z0-9_-]*(?:api[_-]?key|token|secret|password|passwd|pwd|cookie|authorization|client_secret|access[_-]?token|refresh[_-]?token)[A-Za-z0-9_-]*)\s*[:=]\s*([^\s,;]+)"),
    re.compile(r"\bghp_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bAIza[0-9A-Za-z\-_]{20,}\b"),
]
SAFE_COMMAND_PATTERNS = [
    re.compile(r"^\s*git\s+(status|log|diff|show|branch|rev-parse|ls-files)\b", re.IGNORECASE),
    re.compile(r"^\s*npm\s+(test|run\s+(check|test|test:unit|type-check))\b", re.IGNORECASE),
    re.compile(r"^\s*node\s+--check\b", re.IGNORECASE),
    re.compile(r"^\s*python\s+-m\s+py_compile\b", re.IGNORECASE),
    re.compile(r"^\s*(get-childitem|get-content|select-string|ls|dir|pwd|echo|rg|findstr)\b", re.IGNORECASE),
]
LOCAL_HOSTNAMES = {
    "localhost",
    "localhost.localdomain",
    "metadata.google.internal",
    "host.docker.internal",
}
SUCCESS_STATUSES = {"ok", "success"}
WORKSPACE_MUTATION_TOOLS = {
    "fs_write",
    "fs_edit_lines",
    "fs_insert_lines",
    "fs_patch",
    "fs_copy",
    "fs_move",
    "fs_delete",
}
FILE_LIST_TOOLS = {"fs_list"}
FILE_SEARCH_TOOLS = {"fs_search"}
EDIT_ACTIVITY_TOOLS = {"fs_edit_lines", "fs_insert_lines", "fs_patch"}
TEST_RELATED_PATTERN = re.compile(r"\b(test|tests|pytest|unittest|spec|smoke)\b", re.IGNORECASE)
NO_VERIFY_CLAIM_PATTERN = re.compile(
    r"\b(could(?: not|n't) verify (?:the result|the results|completion|success|anything|it)|"
    r"unable to verify (?:the result|the results|completion|success|anything|it)|"
    r"(?:no|not) verified (?:result|results|evidence|completion|success)|"
    r"without verified (?:results|evidence))\b",
    re.IGNORECASE,
)
TEST_PASS_CLAIM_PATTERN = re.compile(
    r"\b(all tests passed|tests passed|test suite passed|tests are green|green test suite)\b",
    re.IGNORECASE,
)
NO_CHANGE_CLAIM_PATTERN = re.compile(
    r"\b(no changes?(?: were)? made|nothing changed|did(?: not|n't) change anything|"
    r"left (?:the )?(?:file|files|workspace|codebase) unchanged)\b",
    re.IGNORECASE,
)
INCOMPLETE_CLAIM_PATTERN = re.compile(
    r"\b(could(?: not|n't) complete|unable to complete|did not complete|was(?: not|n't) able to complete)\b",
    re.IGNORECASE,
)
NO_FILES_FOUND_CLAIM_PATTERN = re.compile(
    r"\b(no (?:files?|entries?|folders|directories) (?:were )?(?:found|returned|listed)|"
    r"found no (?:files?|entries?|folders|directories)|"
    r"did(?: not|n't) find any (?:files?|entries?|folders|directories))\b",
    re.IGNORECASE,
)
NO_SEARCH_RESULTS_CLAIM_PATTERN = re.compile(
    r"\b(no (?:results?|matches?|hits) (?:were )?(?:found|returned)|"
    r"found no (?:results?|matches?|hits)|"
    r"did(?: not|n't) find any (?:results?|matches?|hits)|"
    r"nothing matched)\b",
    re.IGNORECASE,
)
NO_EDITS_APPLIED_CLAIM_PATTERN = re.compile(
    r"\b(no edits?(?: were)? applied|"
    r"did(?: not|n't) edit anything|"
    r"did(?: not|n't) update (?:the )?(?:file|files|workspace|code)|"
    r"nothing was edited|"
    r"no lines? (?:were )?(?:edited|inserted|updated)|"
    r"left (?:the )?(?:file|files) unmodified)\b",
    re.IGNORECASE,
)
NO_PATCH_MATCHES_CLAIM_PATTERN = re.compile(
    r"\b(no matches?(?: were )?found(?: for (?:the )?(?:patch|edit))?|"
    r"did(?: not|n't) find any matches?(?: for (?:the )?(?:patch|edit))?|"
    r"zero matches?(?: for (?:the )?(?:patch|edit))?)\b",
    re.IGNORECASE,
)


def is_path_within_root(path_obj: Path, root_obj: Path) -> bool:
    try:
        path_obj.relative_to(root_obj)
        return True
    except ValueError:
        return False


def resolve_workspace_path(raw_path: str, must_exist: bool = False) -> Path:
    if not raw_path:
        raise ValueError("Path is required.")

    candidate = Path(raw_path)
    if not candidate.is_absolute():
        candidate = (WORKSPACE_ROOT / candidate).resolve()
    else:
        candidate = candidate.resolve()

    if not is_path_within_root(candidate, WORKSPACE_ROOT.resolve()):
        raise ValueError(f"Path is outside workspace root: {candidate}")

    if must_exist and not candidate.exists():
        raise FileNotFoundError(f"Path does not exist: {candidate}")

    return candidate


def to_bool(value, default=False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def is_sensitive_key_name(name: str) -> bool:
    return bool(SENSITIVE_KEY_NAME_PATTERN.search(str(name or "")))


def redact_sensitive_url(value: str) -> str:
    text = str(value or "")
    text = URL_CREDENTIAL_PATTERN.sub(r"\1[REDACTED]@", text)
    return SENSITIVE_QUERY_PARAM_PATTERN.sub(lambda match: f"{match.group(1)}[REDACTED]", text)


def redact_sensitive_text(value: str, max_chars: int = MAX_MEMORY_LOG_CHARS) -> str:
    text = str(value or "")
    stripped = text.strip()

    def _sanitize_json_line(raw_line: str) -> str:
        stripped_line = raw_line.strip()
        if stripped_line[:1] not in {"{", "["}:
            return raw_line
        try:
            structured = json.loads(stripped_line)
            sanitized = json.dumps(redact_sensitive_data(structured, max_chars=max_chars), ensure_ascii=False)
            leading = raw_line[: len(raw_line) - len(raw_line.lstrip())]
            trailing = raw_line[len(raw_line.rstrip()):]
            return f"{leading}{sanitized}{trailing}"
        except Exception:
            return raw_line

    if stripped[:1] in {"{", "["}:
        try:
            structured = json.loads(stripped)
            text = json.dumps(redact_sensitive_data(structured, max_chars=max_chars), ensure_ascii=False)
        except Exception:
            text = "\n".join(_sanitize_json_line(line) for line in text.splitlines())
    elif "\n" in text:
        text = "\n".join(_sanitize_json_line(line) for line in text.splitlines())
    text = redact_sensitive_url(text)
    for pattern in SENSITIVE_VALUE_PATTERNS:
        if pattern.groups >= 2:
            text = pattern.sub(lambda m: f"{m.group(1)}=[REDACTED]", text)
        elif pattern.pattern.lower().startswith("(?i)\\b(bearer"):
            text = pattern.sub("Bearer [REDACTED]", text)
        else:
            text = pattern.sub("[REDACTED]", text)
    if len(text) > max_chars:
        return text[:max_chars] + "\n...[TRUNCATED]"
    return text


def redact_sensitive_data(
    value,
    max_chars: int = MAX_MEMORY_LOG_CHARS,
    max_items: int = 25,
    max_depth: int = 6,
    key_name: str = "",
    _seen=None,
    _depth: int = 0,
):
    if value is None or isinstance(value, (bool, int, float)):
        return value

    if _depth > max_depth:
        return "[TRUNCATED]"

    if is_sensitive_key_name(key_name):
        return "[REDACTED]"

    if isinstance(value, str):
        text = str(value)
        lowered_key = str(key_name or "").lower()
        if any(token in lowered_key for token in {"url", "uri", "link", "href", "location"}):
            text = redact_sensitive_url(text)
        return redact_sensitive_text(text, max_chars=max_chars)

    if isinstance(value, (list, tuple, set)):
        return [
            redact_sensitive_data(
                item,
                max_chars=max_chars,
                max_items=max_items,
                max_depth=max_depth,
                key_name=key_name,
                _seen=_seen,
                _depth=_depth + 1,
            )
            for item in list(value)[:max_items]
        ]

    if isinstance(value, dict):
        seen = _seen if _seen is not None else set()
        value_id = id(value)
        if value_id in seen:
            return "[CIRCULAR]"
        seen.add(value_id)

        result = {}
        items = list(value.items())[:max_items]
        for child_key, child_value in items:
            child_key_text = str(child_key)
            result[child_key_text] = redact_sensitive_data(
                child_value,
                max_chars=max_chars,
                max_items=max_items,
                max_depth=max_depth,
                key_name=child_key_text,
                _seen=seen,
                _depth=_depth + 1,
            )
        return result

    return redact_sensitive_text(str(value), max_chars=max_chars)


def coerce_json_object(value):
    if isinstance(value, dict):
        return value
    text = str(value or "").strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        return {}
    return {}


def _compact_redacted_text(value, max_chars: int = 280) -> str:
    text = redact_sensitive_text(value, max_chars=max_chars)
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _is_success_status(status) -> bool:
    return str(status or "").strip().lower() in SUCCESS_STATUSES


def _coerce_non_negative_int(value):
    try:
        count = int(value)
    except Exception:
        return None
    return count if count >= 0 else None


def _extract_tool_count(payload) -> int | None:
    if not isinstance(payload, dict):
        return None

    explicit_count = _coerce_non_negative_int(payload.get("count"))
    if explicit_count is not None:
        return explicit_count

    for key in ["entries", "matches", "results"]:
        items = payload.get(key)
        if isinstance(items, list):
            return len(items)

    return None


def _extract_edit_match_count(payload) -> int | None:
    if not isinstance(payload, dict):
        return None

    direct_matches = _coerce_non_negative_int(payload.get("matches"))
    if direct_matches is not None:
        return direct_matches

    edit_results = payload.get("edit_results")
    if not isinstance(edit_results, list):
        return None

    total = 0
    found = False
    for item in edit_results:
        if not isinstance(item, dict):
            continue
        match_count = _coerce_non_negative_int(item.get("matches"))
        if match_count is None:
            continue
        total += match_count
        found = True
    return total if found else None


def summarize_tool_outcome(tool_name: str, raw_result, max_summary_chars: int = 280):
    payload = redact_sensitive_data(coerce_json_object(raw_result), max_chars=max_summary_chars * 8)
    safe_tool_name = _compact_redacted_text(tool_name, max_chars=80) or "tool"

    if not payload:
        raw_text = _compact_redacted_text(raw_result, max_chars=max_summary_chars)
        if not raw_text:
            return None
        return {
            "tool_name": safe_tool_name,
            "status": "ok",
            "verified": False,
            "summary": raw_text,
            "artifact_kind": "",
            "next_actions": [],
            "error": "",
        }

    verification = payload.get("verification") if isinstance(payload.get("verification"), dict) else {}
    artifact = payload.get("artifact") if isinstance(payload.get("artifact"), dict) else {}
    status = str(payload.get("status") or payload.get("result", {}).get("status") or "ok").strip().lower() or "ok"
    count = _extract_tool_count(payload)
    changed = payload.get("changed") if isinstance(payload.get("changed"), bool) else None
    wrote = payload.get("wrote") if isinstance(payload.get("wrote"), bool) else None
    dry_run = payload.get("dry_run") if isinstance(payload.get("dry_run"), bool) else None
    applied_matches = _extract_edit_match_count(payload)

    summary_candidates = [
        payload.get("developer_summary"),
        artifact.get("developer_summary"),
        verification.get("reason"),
        payload.get("error"),
        payload.get("reason"),
        payload.get("message"),
    ]
    summary = ""
    for candidate in summary_candidates:
        if candidate in {None, ""}:
            continue
        summary = _compact_redacted_text(candidate, max_chars=max_summary_chars)
        if summary:
            break

    if not summary:
        parts = []
        for key in ["path", "url", "title", "count", "status_code", "selected_index", "executed_steps"]:
            value = payload.get(key)
            if value is None or value == "" or value == []:
                continue
            parts.append(f"{key}={value}")
            if len(parts) >= 3:
                break
        summary = "; ".join(parts) if parts else f"{safe_tool_name} completed."

    next_actions_raw = []
    if isinstance(artifact.get("next_actions"), list):
        next_actions_raw = artifact.get("next_actions")
    elif isinstance(payload.get("next_actions"), list):
        next_actions_raw = payload.get("next_actions")

    next_actions = []
    for item in next_actions_raw[:5]:
        text = _compact_redacted_text(item, max_chars=180)
        if text and text not in next_actions:
            next_actions.append(text)

    return {
        "tool_name": _compact_redacted_text(payload.get("tool") or safe_tool_name, max_chars=80) or safe_tool_name,
        "status": status,
        "verified": isinstance(verification.get("ok"), bool) or bool(artifact),
        "summary": summary,
        "count": count,
        "changed": changed,
        "wrote": wrote,
        "dry_run": dry_run,
        "applied_matches": applied_matches,
        "artifact_kind": _compact_redacted_text(artifact.get("kind", ""), max_chars=80),
        "next_actions": next_actions,
        "error": _compact_redacted_text(payload.get("error", ""), max_chars=180),
    }


def build_execution_grounding_note(user_prompt: str, tool_summaries, max_chars: int = 4000):
    if not isinstance(tool_summaries, list) or not tool_summaries:
        return None

    lines = [
        "[Execution grounding for current turn]",
        f"User goal: {_compact_redacted_text(user_prompt, max_chars=240)}",
        "Answer only from the verified execution facts below.",
        "If something is missing or not verified, say that directly instead of guessing.",
        "Tool outcomes:",
    ]

    aggregated_next = []
    for idx, item in enumerate(tool_summaries[:8], start=1):
        if not isinstance(item, dict):
            continue
        tool_name = _compact_redacted_text(item.get("tool_name", "tool"), max_chars=80) or "tool"
        status = _compact_redacted_text(item.get("status", "unknown"), max_chars=32) or "unknown"
        verified = "yes" if item.get("verified") else "no"
        summary = _compact_redacted_text(item.get("summary", ""), max_chars=220) or f"{tool_name} completed."
        lines.append(f"{idx}. {tool_name} | status={status} | verified={verified} | {summary}")
        for action in item.get("next_actions", [])[:3]:
            text = _compact_redacted_text(action, max_chars=180)
            if text and text not in aggregated_next:
                aggregated_next.append(text)

    if aggregated_next:
        lines.append("Next steps:")
        for idx, action in enumerate(aggregated_next[:5], start=1):
            lines.append(f"- {action}")

    note = "\n".join(lines)
    if len(note) > max_chars:
        note = note[:max_chars] + "\n...[TRUNCATED]"
    return note


def build_response_override(kind: str, reason: str, source: str = "", max_chars: int = 4000):
    normalized_kind = _compact_redacted_text(kind, max_chars=80)
    normalized_reason = _compact_redacted_text(reason, max_chars=max_chars)
    normalized_source = _compact_redacted_text(source, max_chars=80)
    if not normalized_kind and not normalized_reason and not normalized_source:
        return None

    return {
        "source": normalized_source or "grounding-mismatch",
        "kind": normalized_kind or "grounding-mismatch",
        "reason": normalized_reason,
    }


def _resolve_override_source(grounding_override, max_chars: int = 80):
    if not isinstance(grounding_override, dict):
        return ""

    explicit_source = _compact_redacted_text(grounding_override.get("source", ""), max_chars=max_chars)
    if explicit_source:
        return explicit_source

    kind = _compact_redacted_text(grounding_override.get("kind", ""), max_chars=max_chars)
    if kind == "local-access-fallback":
        return "local-access-fallback"
    return "grounding-mismatch"


def _normalize_grounding_override(grounding_mismatch, max_chars: int = 4000):
    if not isinstance(grounding_mismatch, dict):
        return None
    return build_response_override(
        grounding_mismatch.get("kind", ""),
        grounding_mismatch.get("reason", ""),
        source=grounding_mismatch.get("source", ""),
        max_chars=max_chars,
    )


def build_turn_record(user_prompt: str, tool_summaries, final_answer: str, grounding_mismatch=None, max_chars: int = 4000):
    summaries = []
    for item in tool_summaries or []:
        if isinstance(item, dict):
            summaries.append(redact_sensitive_data(item, max_chars=max_chars))

    completed = sum(1 for item in summaries if str(item.get("status", "")).lower() in {"ok", "success"})
    failed = sum(1 for item in summaries if str(item.get("status", "")).lower() not in {"ok", "success"})

    next_steps = []
    for item in summaries:
        for action in item.get("next_actions", [])[:3]:
            text = _compact_redacted_text(action, max_chars=180)
            if text and text not in next_steps:
                next_steps.append(text)

    grounding_override = _normalize_grounding_override(grounding_mismatch, max_chars=max_chars)

    return {
        "saved_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "user_prompt": _compact_redacted_text(user_prompt, max_chars=320),
        "developer_summary": _compact_redacted_text(final_answer, max_chars=600),
        "completed_tools": completed,
        "failed_tools": failed,
        "tool_summaries": summaries[:8],
        "grounding_override": grounding_override,
        "next_steps": next_steps[:5],
    }


def format_session_resume_note(recent_turns, max_chars: int = 4000):
    if not isinstance(recent_turns, list) or not recent_turns:
        return None

    lines = [
        "[Session resume state]",
        "Use this as continuity context for what was already done and what should happen next.",
        "Do not invent progress beyond these saved turn records.",
    ]

    for idx, item in enumerate(recent_turns[-3:], start=1):
        if not isinstance(item, dict):
            continue
        prompt = _compact_redacted_text(item.get("user_prompt", ""), max_chars=180)
        summary = _compact_redacted_text(item.get("developer_summary", ""), max_chars=220)
        grounding_override = item.get("grounding_override") if isinstance(item.get("grounding_override"), dict) else None
        next_steps = item.get("next_steps", []) if isinstance(item.get("next_steps"), list) else []
        lines.append(f"{idx}. Goal: {prompt or 'n/a'}")
        lines.append(f"   Done: {summary or 'n/a'}")
        if grounding_override:
            source = _resolve_override_source(grounding_override)
            kind = _compact_redacted_text(grounding_override.get("kind", ""), max_chars=80) or "grounding-mismatch"
            reason = _compact_redacted_text(grounding_override.get("reason", ""), max_chars=180)
            if reason:
                if kind and kind != source:
                    lines.append(f"   Override: source={source} kind={kind} - {reason}")
                else:
                    lines.append(f"   Override: source={source} - {reason}")
        if next_steps:
            rendered = "; ".join(_compact_redacted_text(step, max_chars=140) for step in next_steps[:3] if step)
            if rendered:
                lines.append(f"   Next: {rendered}")

    note = "\n".join(lines)
    if len(note) > max_chars:
        note = note[:max_chars] + "\n...[TRUNCATED]"
    return note


def collect_recent_next_steps(recent_turns, max_items: int = 5):
    actions = []
    if not isinstance(recent_turns, list):
        return actions

    for item in reversed(recent_turns):
        if not isinstance(item, dict):
            continue
        next_steps = item.get("next_steps", []) if isinstance(item.get("next_steps"), list) else []
        for step in next_steps:
            text = _compact_redacted_text(step, max_chars=180)
            if text and text not in actions:
                actions.append(text)
            if len(actions) >= max_items:
                return actions
    return actions


def format_last_turn_report(recent_turns, max_chars: int = 2500):
    if not isinstance(recent_turns, list) or not recent_turns:
        return None

    latest = recent_turns[-1]
    if not isinstance(latest, dict):
        return None

    lines = [
        "[Last completed turn]",
        f"Goal: {_compact_redacted_text(latest.get('user_prompt', ''), max_chars=220) or 'n/a'}",
        f"Done: {_compact_redacted_text(latest.get('developer_summary', ''), max_chars=320) or 'n/a'}",
        f"Tools: completed={int(latest.get('completed_tools', 0) or 0)} failed={int(latest.get('failed_tools', 0) or 0)}",
    ]

    grounding_override = latest.get("grounding_override") if isinstance(latest.get("grounding_override"), dict) else None
    if grounding_override:
        source = _resolve_override_source(grounding_override)
        kind = _compact_redacted_text(grounding_override.get("kind", ""), max_chars=80) or "grounding-mismatch"
        reason = _compact_redacted_text(grounding_override.get("reason", ""), max_chars=220)
        if reason:
            if kind and kind != source:
                lines.append(f"Grounding override: source={source} kind={kind} | {reason}")
            else:
                lines.append(f"Grounding override: source={source} | {reason}")

    tool_summaries = latest.get("tool_summaries", []) if isinstance(latest.get("tool_summaries"), list) else []
    if tool_summaries:
        lines.append("Verified execution:")
        for item in tool_summaries[:4]:
            if not isinstance(item, dict):
                continue
            tool_name = _compact_redacted_text(item.get("tool_name", "tool"), max_chars=80) or "tool"
            status = _compact_redacted_text(item.get("status", "unknown"), max_chars=32) or "unknown"
            summary = _compact_redacted_text(item.get("summary", ""), max_chars=180) or f"{tool_name} completed."
            lines.append(f"- {tool_name} | status={status} | {summary}")

    next_steps = collect_recent_next_steps([latest], max_items=5)
    if next_steps:
        lines.append("Next steps:")
        for step in next_steps:
            lines.append(f"- {step}")

    note = "\n".join(lines)
    if len(note) > max_chars:
        note = note[:max_chars] + "\n...[TRUNCATED]"
    return note


def detect_final_answer_grounding_mismatch(final_answer: str, tool_summaries):
    base_answer = _compact_redacted_text(final_answer, max_chars=1200)
    if not base_answer:
        return None
    if not isinstance(tool_summaries, list) or not tool_summaries:
        return None

    normalized = []
    for item in tool_summaries[:8]:
        if not isinstance(item, dict):
            continue
        normalized.append(
            {
                "tool_name": str(item.get("tool_name", "")).strip().lower(),
                "status": str(item.get("status", "")).strip().lower(),
                "verified": bool(item.get("verified")),
                "count": _coerce_non_negative_int(item.get("count")),
                "changed": item.get("changed") if isinstance(item.get("changed"), bool) else None,
                "wrote": item.get("wrote") if isinstance(item.get("wrote"), bool) else None,
                "dry_run": item.get("dry_run") if isinstance(item.get("dry_run"), bool) else None,
                "applied_matches": _coerce_non_negative_int(item.get("applied_matches")),
                "summary": _compact_redacted_text(item.get("summary", ""), max_chars=240).lower(),
            }
        )

    if not normalized:
        return None

    lowered = base_answer.lower()
    any_verified = any(item.get("verified") for item in normalized)
    all_success = all(_is_success_status(item.get("status")) for item in normalized)
    mutation_success = any(
        _is_success_status(item.get("status")) and item.get("tool_name") in WORKSPACE_MUTATION_TOOLS
        for item in normalized
    )
    positive_file_list_results = any(
        _is_success_status(item.get("status"))
        and item.get("tool_name") in FILE_LIST_TOOLS
        and (item.get("count") or 0) > 0
        for item in normalized
    )
    positive_file_search_results = any(
        _is_success_status(item.get("status"))
        and item.get("tool_name") in FILE_SEARCH_TOOLS
        and (item.get("count") or 0) > 0
        for item in normalized
    )
    successful_edit_activity = any(
        _is_success_status(item.get("status"))
        and item.get("tool_name") in EDIT_ACTIVITY_TOOLS
        and item.get("changed") is True
        and item.get("dry_run") is not True
        and item.get("wrote") is True
        for item in normalized
    )
    positive_patch_matches = any(
        _is_success_status(item.get("status"))
        and item.get("tool_name") == "fs_patch"
        and (item.get("applied_matches") or 0) > 0
        for item in normalized
    )
    failed_test_related = any(
        not _is_success_status(item.get("status"))
        and TEST_RELATED_PATTERN.search(f"{item.get('tool_name', '')} {item.get('summary', '')}")
        for item in normalized
    )

    if any_verified and NO_VERIFY_CLAIM_PATTERN.search(lowered):
        return {
            "kind": "verification",
            "reason": "The drafted answer says the result was not verified, but verified tool output exists for this turn.",
        }

    if failed_test_related and TEST_PASS_CLAIM_PATTERN.search(lowered):
        return {
            "kind": "tests",
            "reason": "The drafted answer says the tests passed, but verified test-related tool output failed.",
        }

    if mutation_success and NO_CHANGE_CLAIM_PATTERN.search(lowered):
        return {
            "kind": "workspace-change",
            "reason": "The drafted answer says no workspace changes were made, but a workspace mutation tool completed successfully.",
        }

    if positive_file_list_results and NO_FILES_FOUND_CLAIM_PATTERN.search(lowered):
        return {
            "kind": "file-list-results",
            "reason": "The drafted answer says no files were found, but fs_list returned one or more workspace entries.",
        }

    if positive_file_search_results and NO_SEARCH_RESULTS_CLAIM_PATTERN.search(lowered):
        return {
            "kind": "file-search-results",
            "reason": "The drafted answer says no search results were found, but fs_search returned one or more matches.",
        }

    if successful_edit_activity and NO_EDITS_APPLIED_CLAIM_PATTERN.search(lowered):
        return {
            "kind": "edit-activity",
            "reason": "The drafted answer says no edits were applied, but a file edit tool wrote verified changes this turn.",
        }

    if positive_patch_matches and NO_PATCH_MATCHES_CLAIM_PATTERN.search(lowered):
        return {
            "kind": "edit-matches",
            "reason": "The drafted answer says the patch had no matches, but fs_patch reported one or more applied matches.",
        }

    if all_success and INCOMPLETE_CLAIM_PATTERN.search(lowered):
        return {
            "kind": "completion",
            "reason": "The drafted answer says the task was not completed, but all captured tool results for this turn succeeded.",
        }

    return None


def _build_grounded_final_answer(tool_summaries, max_chars: int = 1800):
    success_count = 0
    failed_count = 0
    for item in tool_summaries or []:
        if not isinstance(item, dict):
            continue
        if _is_success_status(item.get("status")):
            success_count += 1
        else:
            failed_count += 1

    if success_count or failed_count:
        if failed_count:
            intro = (
                "The drafted response did not match the verified execution results, so this summary is grounded directly in tool output. "
                f"Tool results for this turn: {success_count} succeeded and {failed_count} failed."
            )
        else:
            intro = (
                "The drafted response did not match the verified execution results, so this summary is grounded directly in tool output. "
                f"Tool results for this turn: {success_count} succeeded."
            )
    else:
        intro = "The drafted response did not match the verified execution results, so this summary is grounded directly in tool output."

    return augment_final_answer_with_grounding(intro, tool_summaries, max_chars=max_chars)


def reconcile_final_answer_with_grounding(final_answer: str, tool_summaries, max_chars: int = 1800):
    mismatch = detect_final_answer_grounding_mismatch(final_answer, tool_summaries)
    if mismatch:
        return _build_grounded_final_answer(tool_summaries, max_chars=max_chars)
    return augment_final_answer_with_grounding(final_answer, tool_summaries, max_chars=max_chars)


def reconcile_final_answer_with_grounding_metadata(final_answer: str, tool_summaries, max_chars: int = 1800):
    mismatch = detect_final_answer_grounding_mismatch(final_answer, tool_summaries)
    if mismatch:
        answer = _build_grounded_final_answer(tool_summaries, max_chars=max_chars)
    else:
        answer = augment_final_answer_with_grounding(final_answer, tool_summaries, max_chars=max_chars)

    return {
        "answer": answer,
        "grounding_mismatch": build_response_override(
            mismatch.get("kind", ""),
            mismatch.get("reason", ""),
            source="grounding-mismatch",
            max_chars=max_chars,
        ) if isinstance(mismatch, dict) else None,
    }


def augment_final_answer_with_grounding(final_answer: str, tool_summaries, max_chars: int = 1800):
    base_answer = str(final_answer or "").strip()
    if not isinstance(tool_summaries, list) or not tool_summaries:
        return base_answer

    lowered = base_answer.lower()
    if "verified execution:" in lowered and "next steps:" in lowered:
        return base_answer

    lines = []
    for item in tool_summaries[:4]:
        if not isinstance(item, dict):
            continue
        tool_name = _compact_redacted_text(item.get("tool_name", "tool"), max_chars=80) or "tool"
        status = _compact_redacted_text(item.get("status", "unknown"), max_chars=32) or "unknown"
        summary = _compact_redacted_text(item.get("summary", ""), max_chars=180) or f"{tool_name} completed."
        lines.append(f"- {tool_name} | status={status} | {summary}")

    if not lines:
        return base_answer

    next_steps = []
    for item in tool_summaries:
        if not isinstance(item, dict):
            continue
        for step in item.get("next_actions", [])[:3]:
            text = _compact_redacted_text(step, max_chars=160)
            if text and text not in next_steps:
                next_steps.append(text)
            if len(next_steps) >= 4:
                break
        if len(next_steps) >= 4:
            break

    addendum = ["Verified execution:"] + lines
    if next_steps:
        addendum.append("Next steps:")
        addendum.extend(f"- {step}" for step in next_steps)

    merged = base_answer + "\n\n" + "\n".join(addendum)
    if len(merged) > max_chars:
        merged = merged[:max_chars] + "\n...[TRUNCATED]"
    return merged


POWERSHELL_METACHAR_PATTERNS = [
    re.compile(r"\$\("),        # subexpression operator
    re.compile(r"@\("),          # array subexpression
    re.compile(r"`"),             # backtick escape character
    re.compile(r"&\s*\{"),       # script block invocation
    re.compile(r"\.\.[\\/]"),   # parent directory traversal
    re.compile(r"--%"),           # stop-parsing token
    re.compile(r"Invoke-Expression|iex\b", re.IGNORECASE),
]


def run_command_is_safe_in_restricted_mode(command: str) -> bool:
    trimmed = str(command or "").strip()
    if not trimmed:
        return False
    if any(token in trimmed for token in ["&&", "||", ";", "|"]):
        return False
    if any(pattern.search(trimmed) for pattern in POWERSHELL_METACHAR_PATTERNS):
        return False
    return any(pattern.search(trimmed) for pattern in SAFE_COMMAND_PATTERNS)


def is_private_or_local_host(hostname: str) -> bool:
    host = (hostname or "").strip().lower()
    if not host:
        return True
    if host in LOCAL_HOSTNAMES or host.endswith(".local"):
        return True

    try:
        ip_obj = ipaddress.ip_address(host)
        return (
            ip_obj.is_private
            or ip_obj.is_loopback
            or ip_obj.is_link_local
            or ip_obj.is_multicast
            or ip_obj.is_reserved
            or ip_obj.is_unspecified
        )
    except ValueError:
        pass

    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        return False
    except Exception:
        return True

    for item in infos:
        try:
            resolved_ip = item[4][0]
            ip_obj = ipaddress.ip_address(resolved_ip)
            if (
                ip_obj.is_private
                or ip_obj.is_loopback
                or ip_obj.is_link_local
                or ip_obj.is_multicast
                or ip_obj.is_reserved
                or ip_obj.is_unspecified
            ):
                return True
        except Exception:
            continue
    return False
