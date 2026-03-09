import hashlib
import hmac
import json
import secrets
import time


_SECRET = secrets.token_bytes(32)
_DEFAULT_TTL_SEC = 600


def _normalized_payload(tool_name: str, arguments: dict) -> str:
    safe_args = dict(arguments or {})
    safe_args.pop("confirm", None)
    safe_args.pop("confirm_token", None)
    return json.dumps(
        {
            "tool_name": str(tool_name or "").strip(),
            "arguments": safe_args,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def issue_confirm_token(tool_name: str, arguments: dict, issued_at: int | None = None) -> str:
    ts = int(time.time() if issued_at is None else issued_at)
    payload = f"{ts}|{_normalized_payload(tool_name, arguments)}"
    digest = hmac.new(_SECRET, payload.encode("utf-8"), hashlib.sha256).hexdigest()[:24]
    return f"{ts}:{digest}"


def validate_confirm_token(tool_name: str, arguments: dict, token: str, ttl_sec: int = _DEFAULT_TTL_SEC) -> bool:
    raw = str(token or "").strip()
    if not raw or ":" not in raw:
        return False
    ts_text, _signature = raw.split(":", 1)
    try:
        ts = int(ts_text)
    except Exception:
        return False
    now = int(time.time())
    if ts > now or (now - ts) > max(1, int(ttl_sec)):
        return False
    expected = issue_confirm_token(tool_name, arguments, issued_at=ts)
    return hmac.compare_digest(expected, raw)
