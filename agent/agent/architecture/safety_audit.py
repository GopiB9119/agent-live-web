import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Callable


SAFETY_AUDIT_ENABLED_ENV = "AGENT_SAFETY_AUDIT_ENABLED"
SAFETY_AUDIT_FILE_ENV = "AGENT_SAFETY_AUDIT_FILE"


def _to_bool(value, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _default_file_path(workspace_root: Path) -> Path:
    override = str(os.getenv(SAFETY_AUDIT_FILE_ENV, "")).strip()
    if override:
        return Path(override)
    return workspace_root / ".agent-state" / "safety-events.jsonl"


def write_safety_event(
    workspace_root: Path,
    event: dict[str, Any],
    redact_fn: Callable[[str], str] | None = None,
):
    if not _to_bool(os.getenv(SAFETY_AUDIT_ENABLED_ENV, "1"), True):
        return

    target = _default_file_path(Path(workspace_root).resolve())
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        return False

    payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "user_mode": str(os.getenv("AGENT_INTERFACE_MODE", "python")).strip().lower() or "python",
        **(event or {}),
    }

    if redact_fn:
        for key in ("arguments_summary", "result_summary", "error"):
            if key in payload and payload[key] is not None:
                payload[key] = redact_fn(str(payload[key]))

    try:
        with target.open("a", encoding="utf-8", errors="replace", newline="") as fh:
            fh.write(json.dumps(payload, ensure_ascii=True) + "\n")
    except OSError:
        return False

    return True
