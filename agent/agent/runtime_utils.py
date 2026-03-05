import os
import re
import socket
import ipaddress
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
SENSITIVE_VALUE_PATTERNS = [
    re.compile(r"(?i)\b(bearer)\s+[A-Za-z0-9._\-]{12,}"),
    re.compile(r"(?i)\b(api[_-]?key|token|secret|password|passwd|pwd|cookie|authorization)\s*[:=]\s*([^\s,;]+)"),
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


def redact_sensitive_text(value: str, max_chars: int = MAX_MEMORY_LOG_CHARS) -> str:
    text = str(value or "")
    for pattern in SENSITIVE_VALUE_PATTERNS:
        if pattern.pattern.lower().startswith("(?i)\\b(api"):
            text = pattern.sub(lambda m: f"{m.group(1)}=[REDACTED]", text)
        elif pattern.pattern.lower().startswith("(?i)\\b(bearer"):
            text = pattern.sub("Bearer [REDACTED]", text)
        else:
            text = pattern.sub("[REDACTED]", text)
    if len(text) > max_chars:
        return text[:max_chars] + "\n...[TRUNCATED]"
    return text


def run_command_is_safe_in_restricted_mode(command: str) -> bool:
    trimmed = str(command or "").strip()
    if not trimmed:
        return False
    if any(token in trimmed for token in ["&&", "||", ";", "|"]):
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
