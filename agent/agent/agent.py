import os
import json
import asyncio
import re
import difflib
import sys
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from openai import AzureOpenAI, OpenAI
from tools import (
    AGENT_TOOLS,
    AVAILABLE_FUNCTIONS,
    execute_tool_with_policy,
    get_mcp_runtime_status,
    init_mcp_client,
    shutdown_mcp_client,
)
try:
    from task_spec import build_task_spec
except Exception:
    from .task_spec import build_task_spec
try:
    from capability_router import (
        activate_turn_tool_context,
        build_turn_tool_context as resolve_turn_tool_context,
        reset_turn_tool_context,
        summarize_turn_tool_context,
    )
except Exception:
    from .capability_router import (
        activate_turn_tool_context,
        build_turn_tool_context as resolve_turn_tool_context,
        reset_turn_tool_context,
        summarize_turn_tool_context,
    )

# Load environment variables
load_dotenv()


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name, "true" if default else "false")
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except Exception:
        return default


AGENT_FAST_MODE = _env_flag("AGENT_FAST_MODE", True)
MAX_ITERATIONS = max(2, min(_env_int("AGENT_MAX_ITERATIONS", 7 if AGENT_FAST_MODE else 10), 40))
MAX_HISTORY_MESSAGES = max(20, min(_env_int("AGENT_MAX_HISTORY_MESSAGES", 40 if AGENT_FAST_MODE else 60), 200))
TOOL_TIMEOUT_SEC = float(os.environ.get("AGENT_TOOL_TIMEOUT_SEC", "90" if AGENT_FAST_MODE else "180"))
MEMORY_AUTO_LOG = _env_flag("AGENT_MEMORY_AUTO_LOG", False)
MEMORY_PRIVATE_SESSION = _env_flag("AGENT_PRIVATE_SESSION", True)
MAX_MEMORY_LOG_CHARS = max(600, min(_env_int("AGENT_MEMORY_LOG_MAX_CHARS", 4000), 50000))
MEMORY_AUTO_RECALL = _env_flag("AGENT_MEMORY_AUTO_RECALL", True)
MEMORY_RECALL_TOP_K = max(1, min(_env_int("AGENT_MEMORY_RECALL_TOP_K", 2 if AGENT_FAST_MODE else 4), 12))
MEMORY_RECALL_DAYS_BACK = max(1, min(_env_int("AGENT_MEMORY_RECALL_DAYS_BACK", 30), 180))
MEMORY_RECALL_MAX_CHARS = max(600, min(_env_int("AGENT_MEMORY_RECALL_MAX_CHARS", 2200 if AGENT_FAST_MODE else 3500), 30000))
SESSION_STATE_ENABLED = _env_flag("AGENT_SESSION_STATE_ENABLED", True)
SESSION_STATE_MAX_MESSAGES = max(8, min(_env_int("AGENT_SESSION_STATE_MAX_MESSAGES", 80), 300))
REPO_ROOT = Path(__file__).resolve().parents[2]
SESSION_STATE_FILE = Path(
    os.environ.get(
        "AGENT_SESSION_STATE_FILE",
        str(REPO_ROOT / ".agent-state" / "last_session.json"),
    )
)
ENV_FILE = Path(os.environ.get("AGENT_ENV_FILE", str(REPO_ROOT / ".env")))
ENV_EXAMPLE_FILE = Path(os.environ.get("AGENT_ENV_EXAMPLE_FILE", str(REPO_ROOT / ".env.example")))
STARTUP_STATE_FILE = Path(
    os.environ.get(
        "AGENT_STARTUP_STATE_FILE",
        str(REPO_ROOT / ".agent-state" / "startup_state.json"),
    )
)


def _create_client_and_model():
    provider_pref = str(os.environ.get("AGENT_PROVIDER", "auto")).strip().lower()
    openai_key = str(os.environ.get("OPENAI_API_KEY", "")).strip()
    azure_key = str(os.environ.get("azure_key", "")).strip()
    azure_endpoint = str(os.environ.get("azure_endpoint_uri", "")).strip()

    wants_openai = provider_pref in {"openai", "codex"} or (provider_pref == "auto" and openai_key)
    if wants_openai:
        if not openai_key:
            return None, str(os.environ.get("AGENT_MODEL", "codex-5.3")), "openai", "OPENAI_API_KEY is missing."
        model_name = str(os.environ.get("AGENT_MODEL", os.environ.get("OPENAI_MODEL", "codex-5.3"))).strip() or "codex-5.3"
        return OpenAI(api_key=openai_key), model_name, "openai", ""

    if not azure_key or not azure_endpoint:
        missing = []
        if not azure_key:
            missing.append("azure_key")
        if not azure_endpoint:
            missing.append("azure_endpoint_uri")
        missing_text = ", ".join(missing) if missing else "azure credentials"
        return None, str(os.environ.get("azure_deployment_name", "gpt-4o")), "azure", f"Missing {missing_text}."

    model_name = str(os.environ.get("AGENT_MODEL", os.environ.get("azure_deployment_name", "gpt-4o"))).strip() or "gpt-4o"
    return (
        AzureOpenAI(
            api_key=azure_key,
            api_version=os.environ.get("azure_api_version", "2024-12-01-preview"),
            azure_endpoint=azure_endpoint,
        ),
        model_name,
        "azure",
        "",
    )


client, MODEL, MODEL_PROVIDER, MODEL_SETUP_ERROR = _create_client_and_model()
SENSITIVE_MEMORY_PATTERNS = [
    re.compile(r"(?i)\b(bearer)\s+[A-Za-z0-9._\-]{12,}"),
    re.compile(r"(?i)\b(api[_-]?key|token|secret|password|passwd|pwd|cookie|authorization)\s*[:=]\s*([^\s,;]+)"),
    re.compile(r"\bghp_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bAIza[0-9A-Za-z\-_]{20,}\b"),
]

RUNTIME_EXECUTION_GUIDE = (
    "[Runtime execution guide]\n"
    "- For broad goals, prefer task_autopilot first to map files and next actions.\n"
    "- For deterministic multi-step execution, use workflow_execute with explicit per-step verification.\n"
    "- For edits, prefer fs tools and verify with fs_read/fs_search before final response.\n"
    "- When tool output is ambiguous, call the tool_catalog or a validation tool instead of guessing."
)


def _redact_for_memory(value: str) -> str:
    text = str(value or "")
    for pattern in SENSITIVE_MEMORY_PATTERNS:
        if pattern.pattern.lower().startswith("(?i)\\b(api"):
            text = pattern.sub(lambda m: f"{m.group(1)}=[REDACTED]", text)
        elif pattern.pattern.lower().startswith("(?i)\\b(bearer"):
            text = pattern.sub("Bearer [REDACTED]", text)
        else:
            text = pattern.sub("[REDACTED]", text)
    if len(text) > MAX_MEMORY_LOG_CHARS:
        text = text[:MAX_MEMORY_LOG_CHARS] + "\n...[TRUNCATED]"
    return text


def _load_system_prompt() -> str:
    prompt_file = Path(__file__).resolve().with_name("SYSTEM_PROMPT.md")
    default_prompt = (
        "You are an autonomous Live Web + Workspace Automation Agent. "
        "Use browser tools for interactive web tasks and fs tools for local codebase tasks. "
        "Never claim local folders are inaccessible when fs tools are available."
    )
    try:
        if prompt_file.exists():
            content = prompt_file.read_text(encoding="utf-8", errors="replace").strip()
            if content:
                return content
    except Exception:
        pass
    return default_prompt


def _format_model_setup_help() -> str:
    return (
        "Model configuration is missing.\n"
        "OpenAI example:\n"
        "  AGENT_PROVIDER=openai\n"
        "  OPENAI_API_KEY=...\n"
        "  AGENT_MODEL=codex-5.3\n"
        "Azure example:\n"
        "  AGENT_PROVIDER=azure\n"
        "  azure_key=...\n"
        "  azure_endpoint_uri=...\n"
        "  azure_deployment_name=...\n"
        "Docs:\n"
        "  - README.md\n"
        "  - docs/MODES.md\n"
        "  - docs/TROUBLESHOOTING.md"
    )


def _format_startup_banner() -> str:
    return (
        "Agent Live Web - Python Agent Mode\n"
        f"- provider: {MODEL_PROVIDER}\n"
        f"- model: {MODEL}\n"
        f"- safety_mode: shared preview/confirm/block policy\n"
        f"- session_state: {'enabled' if SESSION_STATE_ENABLED else 'disabled'}\n"
        f"- memory_recall: {'enabled' if MEMORY_AUTO_RECALL else 'disabled'}\n"
        f"- baseline_check: npm run verify\n"
        "- help: /help | /status | /mcp | /doctor | /setup"
    )


def _format_mcp_startup_note(status: dict) -> str:
    if status.get("connected"):
        note = (
            f"[MCP] Browser tool surface connected ({status.get('tool_count', 0)} tools).\n"
            "Repo/workspace tools and browser tools are both available."
        )
        proxy_lines = _format_mcp_proxy_status_lines(status)
        proxy_summary = next((line for line in proxy_lines if line.startswith("- mcp_proxy_summary: ")), "")
        if proxy_summary:
            note += f"\n{proxy_summary.replace('- mcp_proxy_summary: ', '[MCP] Proxy: ', 1)}"
        return note
    error = str(status.get("error", "") or "unknown startup error")
    return (
        "[MCP] Browser tool surface is unavailable for this run.\n"
        "Repo/workspace tools still work.\n"
        f"Last error: {error}\n"
        "Next steps:\n"
        "  - direct launcher: npm run agent:vscode\n"
        "  - raw probe: npm run mcp:probe -- --target=launcher --profile=minimal --step=tools-list\n"
        "  - docs: docs/TROUBLESHOOTING.md"
    )


def _tool_result_status(raw_response) -> str:
    if isinstance(raw_response, dict):
        return str(raw_response.get("status", "ok") or "ok").strip().lower()
    try:
        parsed = json.loads(str(raw_response))
        if isinstance(parsed, dict):
            return str(parsed.get("status", "ok") or "ok").strip().lower()
    except Exception:
        pass
    return "ok"


def _trusted_startup_reason_for_turn(turn_used_tools: bool, tool_results: list) -> str | None:
    if not turn_used_tools:
        return "direct_answer"
    statuses = [_tool_result_status(item) for item in list(tool_results or [])]
    if statuses and all(status in {"ok", "success"} for status in statuses):
        return "tool_assisted_answer"
    return None


def _describe_resume_state(resume_guard: dict, startup_state: dict) -> str:
    if not SESSION_STATE_ENABLED:
        return "disabled"
    if bool(resume_guard.get("allowed")):
        return "ready for auto-resume"
    reason = str(resume_guard.get("reason", "") or "").strip().lower()
    startup_completed = bool(startup_state.get("startup_completed", False))
    completion_reason = str(startup_state.get("startup_completion_reason", "") or "").strip().lower()
    if reason in {"no resume", "resume not allowed"}:
        if not startup_completed:
            return "locked until first successful turn"
        return "locked"
    if reason == "local model config is not ready yet":
        return "locked until model config is ready"
    if reason == "complete one successful startup before auto-resume":
        if completion_reason == "none" or not completion_reason:
            return "locked until first successful turn"
        return f"locked until a trusted startup completes ({completion_reason})"
    if reason:
        return f"locked ({reason})"
    return "locked"


def _format_startup_used_tools(startup_state: dict) -> str:
    if not bool(startup_state.get("startup_completed", False)):
        return "none"
    return str(bool(startup_state.get("startup_used_tools", False)))


def _describe_startup_trust(startup_state: dict) -> str:
    if not bool(startup_state.get("startup_completed", False)):
        return "not yet earned"
    reason = str(startup_state.get("startup_completion_reason", "") or "").strip().lower()
    if reason == "direct_answer":
        return "direct answer"
    if reason == "tool_assisted_answer":
        return "tool-assisted verified"
    return "earned"


def _format_mcp_proxy_status_lines(mcp_status: dict) -> list[str]:
    proxy = mcp_status.get("proxy_status")
    if not isinstance(proxy, dict) or not proxy:
        return []

    ownership = proxy.get("ownership") if isinstance(proxy.get("ownership"), dict) else {}
    owner = str(ownership.get("owner", "") or "unknown")
    active_owner = str(ownership.get("active_owner", "") or owner)
    return [
        f"- mcp_proxy_runtime_status: {proxy.get('runtime_status', 'unknown')}",
        f"- mcp_proxy_startup_trust: {proxy.get('startup_trust', '') or 'none'}",
        f"- mcp_proxy_resume_state: {proxy.get('resume_state', '') or 'none'}",
        f"- mcp_proxy_owner: {owner} (active={active_owner})",
        f"- mcp_proxy_summary: {proxy.get('summary', '') or proxy.get('trust_summary', '') or 'none'}",
    ]


def _build_mcp_proxy_status_payload(mcp_status=None) -> dict:
    mcp = mcp_status or get_mcp_runtime_status()
    proxy = mcp.get("proxy_status") if isinstance(mcp.get("proxy_status"), dict) else {}
    proxy_payload = dict(proxy) if proxy else {
        "runtime_status": "unavailable",
        "summary": "direct MCP proxy status is unavailable until MCP connects",
    }
    return {
        "status": "ok",
        "report": "mcp_proxy_status",
        "mode": "python_agent",
        "mcp": {
            "status": mcp.get("status", "unknown"),
            "connected": bool(mcp.get("connected", False)),
            "tool_count": int(mcp.get("tool_count", 0) or 0),
            "error": str(mcp.get("error", "") or ""),
        },
        "proxy": proxy_payload,
    }


def _format_mcp_proxy_status_report(mcp_status=None) -> str:
    payload = _build_mcp_proxy_status_payload(mcp_status)
    mcp = payload["mcp"]
    proxy = payload["proxy"]
    lines = [
        "MCP Proxy Status",
        f"- mcp_status: {mcp.get('status', 'unknown')}",
        f"- mcp_connected: {mcp.get('connected', False)}",
        f"- mcp_tool_count: {mcp.get('tool_count', 0)}",
        f"- mcp_error: {mcp.get('error', '') or 'none'}",
    ]
    if proxy.get("runtime_status") != "unavailable" or proxy.get("summary"):
        owner_info = proxy.get("ownership") if isinstance(proxy.get("ownership"), dict) else {}
        owner = str(owner_info.get("owner", "") or "unknown")
        active_owner = str(owner_info.get("active_owner", "") or owner)
        lines.extend(
            [
                f"- mcp_proxy_runtime_status: {proxy.get('runtime_status', 'unknown')}",
                f"- mcp_proxy_startup_trust: {proxy.get('startup_trust', '') or 'none'}",
                f"- mcp_proxy_resume_state: {proxy.get('resume_state', '') or 'none'}",
                f"- mcp_proxy_owner: {owner} (active={active_owner})",
                f"- mcp_proxy_summary: {proxy.get('summary', '') or proxy.get('trust_summary', '') or 'none'}",
            ]
        )
    else:
        lines.append("- mcp_proxy_runtime_status: unavailable")
        lines.append("- mcp_proxy_summary: direct MCP proxy status is unavailable until MCP connects")
    return "\n".join(lines)


def _format_preflight_report(mcp_status=None) -> str:
    mcp = mcp_status or get_mcp_runtime_status()
    startup_state = _load_startup_state()
    resume_guard = _session_resume_guard(startup_state=startup_state)
    model_ready = client is not None
    env_exists = ENV_FILE.exists()
    env_example_exists = ENV_EXAMPLE_FILE.exists()
    browser_tools_registered = len([name for name in AVAILABLE_FUNCTIONS if name.startswith("browser_")])
    browser_tools_usable = bool(mcp.get("connected") and browser_tools_registered > 0)

    lines = [
        "Agent Preflight",
        "- mode: python",
        f"- env_file: {ENV_FILE} (exists={env_exists})",
        f"- env_example_file: {ENV_EXAMPLE_FILE} (exists={env_example_exists})",
        f"- startup_state_file: {STARTUP_STATE_FILE}",
        f"- startup_completed: {bool(startup_state.get('startup_completed', False))}",
        f"- startup_completion_reason: {startup_state.get('startup_completion_reason', '') or 'none'}",
        f"- startup_used_tools: {_format_startup_used_tools(startup_state)}",
        f"- startup_trust: {_describe_startup_trust(startup_state)}",
        f"- successful_startup_at: {startup_state.get('successful_startup_at', '') or 'none'}",
        f"- successful_startup_count: {int(startup_state.get('successful_startup_count', 0) or 0)}",
        f"- provider: {MODEL_PROVIDER}",
        f"- model: {MODEL}",
        f"- model_ready: {model_ready}",
        f"- model_setup_error: {MODEL_SETUP_ERROR or 'none'}",
        f"- mcp_status: {mcp.get('status', 'unknown')}",
        f"- mcp_connected: {mcp.get('connected', False)}",
        f"- mcp_tool_count: {mcp.get('tool_count', 0)}",
        f"- mcp_error: {mcp.get('error', '') or 'none'}",
        f"- browser_tools_registered: {browser_tools_registered}",
        f"- browser_tools_usable: {browser_tools_usable}",
        f"- resume_state: {_describe_resume_state(resume_guard, startup_state)}",
        f"- session_auto_resume_ready: {resume_guard.get('allowed', False)}",
        f"- session_resume_reason: {resume_guard.get('reason', '') or 'none'}",
        f"- baseline_check: npm run verify",
    ]
    lines.extend(_format_mcp_proxy_status_lines(mcp))

    lines.append("Next steps:")
    if not env_exists and env_example_exists and not model_ready:
        lines.append("- create local config: copy .env.example .env")
    elif not env_exists and env_example_exists and model_ready:
        lines.append("- optional local config: copy .env.example .env if you want repo-local settings")
    if not model_ready:
        lines.append("- fill the provider block you want to use in .env")
        lines.append("- start with the examples shown in README.md")
    else:
        lines.append("- start interactive mode: npm run agent:python")
    if not mcp.get("connected", False):
        lines.append("- direct MCP/browser launch: npm run agent:vscode")
        lines.append("- raw probe: npm run mcp:probe -- --target=launcher --profile=minimal --step=tools-list")
    lines.append("- docs: docs/MODES.md | docs/TROUBLESHOOTING.md")
    return "\n".join(lines)


def _build_turn_tool_context(user_text: str, mcp_status=None) -> dict:
    status = mcp_status or get_mcp_runtime_status()
    task_spec = build_task_spec(user_text)
    return resolve_turn_tool_context(
        task_spec=task_spec,
        agent_tools=AGENT_TOOLS,
        available_tool_names=AVAILABLE_FUNCTIONS.keys(),
        mcp_status=status,
    )


def _wants_preflight_mode() -> bool:
    lowered = {str(arg).strip().lower() for arg in sys.argv[1:]}
    return "--preflight" in lowered or "--setup" in lowered


def _wants_mcp_status_mode() -> bool:
    lowered = {str(arg).strip().lower() for arg in sys.argv[1:]}
    return "--mcp-status" in lowered or "--proxy-status" in lowered


def _wants_json_output() -> bool:
    lowered = {str(arg).strip().lower() for arg in sys.argv[1:]}
    return "--json" in lowered


def _looks_like_local_access_request(text: str) -> bool:
    lowered = text.lower()
    has_path_hint = bool(re.search(r"[a-zA-Z]:\\", text)) or "/" in text or "\\" in text
    has_scope_hint = any(word in lowered for word in ["codebase", "folder", "directory", "repo", "repository", "file", "path"])
    has_action_hint = any(word in lowered for word in ["see", "read", "open", "inspect", "check", "scan", "list", "show", "analyze"])
    return (has_path_hint and has_scope_hint) or (has_scope_hint and has_action_hint)


def _looks_like_local_access_refusal(text: str) -> bool:
    lowered = (text or "").lower()
    refusal_phrases = [
        "can't directly access",
        "cannot directly access",
        "can't access folders on your local",
        "cannot access folders on your local",
        "don't have access to your local",
        "do not have access to your local",
        "can't access your local computer",
        "can't directly interact with your local",
        "cannot directly interact with your local",
        "virtual workspace",
        "upload the files",
        "upload the necessary files",
    ]
    if any(phrase in lowered for phrase in refusal_phrases):
        return True
    return bool(
        re.search(r"\b(can('|no)t|cannot)\b.*\b(local|folder|directory|file system|computer)\b", lowered)
    )


def _extract_path_candidates(user_text: str):
    # Start from the first Windows drive-like sequence and progressively trim trailing words.
    match = re.search(r"[a-zA-Z]:\\", user_text)
    if not match:
        return []

    tail = user_text[match.start():].strip()
    tail = tail.strip("\"'`").strip()
    tail = tail.rstrip(".,;:!?")
    if not tail:
        return []

    candidates = []
    words = tail.split()
    for i in range(len(words), 0, -1):
        candidate = " ".join(words[:i]).strip()
        candidate = candidate.strip("\"'`").rstrip(".,;:!?")
        if candidate and candidate not in candidates:
            candidates.append(candidate)
    return candidates


async def _auto_local_access_fallback(user_prompt: str) -> str:
    fs_list_fn = AVAILABLE_FUNCTIONS.get("fs_list")
    fs_read_fn = AVAILABLE_FUNCTIONS.get("fs_read")
    if not fs_list_fn or not fs_read_fn:
        return "Local tools are not available in this session."

    path_candidates = _extract_path_candidates(user_prompt)
    if not path_candidates:
        path_candidates = ["."]

    for candidate in path_candidates:
        try:
            listing_raw = await fs_list_fn({"path": candidate, "recursive": False, "max_entries": 40, "include_hidden": False})
            listing = json.loads(listing_raw)
            if listing.get("status") == "ok":
                entries = listing.get("entries", [])
                head = entries[:12]
                rendered = "\n".join(f"- {item.get('path')} ({item.get('type')})" for item in head)
                return (
                    f"I can access your local workspace. I inspected `{candidate}`.\n"
                    f"Found {listing.get('count', 0)} entries.\n"
                    f"{rendered if rendered else '- (empty directory)'}\n"
                    "Tell me if you want recursive scan, specific file reads, or full codebase summary."
                )
        except Exception:
            pass

        try:
            read_raw = await fs_read_fn({"path": candidate, "max_chars": 4000})
            read_result = json.loads(read_raw)
            if read_result.get("status") == "ok":
                preview = read_result.get("content", "")
                return (
                    f"I can access your local workspace. I read `{candidate}`.\n"
                    f"Preview:\n{preview[:1200]}"
                )
        except Exception:
            pass

    attempted = ", ".join(f"`{p}`" for p in path_candidates[:5])
    return f"I attempted local access for {attempted}, but could not resolve a valid path. Share the exact folder path and I will inspect it."


async def _preflight_local_context(user_prompt: str):
    if not _looks_like_local_access_request(user_prompt):
        return None
    summary = await _auto_local_access_fallback(user_prompt)
    return summary


async def _memory_bootstrap_context():
    memory_bootstrap_fn = AVAILABLE_FUNCTIONS.get("memory_bootstrap")
    if not memory_bootstrap_fn:
        return None
    try:
        raw = await memory_bootstrap_fn(
            {
                "include_long_term": MEMORY_PRIVATE_SESSION,
                "max_chars": 24000,
            }
        )
        parsed = json.loads(raw)
        if parsed.get("status") == "ok" and parsed.get("content"):
            return parsed
    except Exception:
        return None
    return None


async def _memory_log_event(content: str, role: str, importance: int = 3, tags=None):
    if not MEMORY_AUTO_LOG:
        return
    memory_log_fn = AVAILABLE_FUNCTIONS.get("memory_log")
    if not memory_log_fn:
        return
    sanitized = _redact_for_memory(content).strip()
    if not sanitized:
        return
    try:
        await memory_log_fn(
            {
                "content": sanitized,
                "role": role,
                "importance": int(importance),
                "tags": tags or [],
            }
        )
    except Exception:
        # Do not block conversation on memory log failures.
        pass


def _parse_tool_arguments(raw_args):
    if raw_args is None:
        return {}
    if isinstance(raw_args, dict):
        return raw_args

    text = str(raw_args).strip()
    if not text:
        return {}

    candidates = [text]
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        candidates.insert(0, fenced.group(1).strip())

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            continue
    return {}


def _normalize_tool_arguments(function_args):
    if not isinstance(function_args, dict):
        return {}
    # Allow wrapper payloads like {"arguments": {...}} from future model/tool adapters.
    if set(function_args.keys()) == {"arguments"} and isinstance(function_args.get("arguments"), dict):
        return function_args["arguments"]
    return function_args


def _resolve_tool_callable(function_name: str):
    requested = str(function_name or "").strip()
    if not requested:
        return "", None, []

    direct = AVAILABLE_FUNCTIONS.get(requested)
    if direct:
        return requested, direct, []

    lower_map = {name.lower(): name for name in AVAILABLE_FUNCTIONS}
    lower_hit = lower_map.get(requested.lower())
    if lower_hit:
        return lower_hit, AVAILABLE_FUNCTIONS.get(lower_hit), [lower_hit]

    choices = sorted(AVAILABLE_FUNCTIONS.keys())
    close = difflib.get_close_matches(requested, choices, n=5, cutoff=0.55)
    if close:
        best = close[0]
        return best, AVAILABLE_FUNCTIONS.get(best), close

    return requested, None, []


def _normalize_saved_chat_message(item):
    if not isinstance(item, dict):
        return None
    role = str(item.get("role", "")).strip().lower()
    content = item.get("content")
    if role not in {"user", "assistant"}:
        return None
    if not isinstance(content, str) or not content.strip():
        return None
    return {"role": role, "content": content.strip()}


def _load_startup_state():
    try:
        if not STARTUP_STATE_FILE.exists():
            return {}
        payload = json.loads(STARTUP_STATE_FILE.read_text(encoding="utf-8", errors="replace"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _save_startup_state(payload):
    try:
        STARTUP_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        STARTUP_STATE_FILE.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    except Exception:
        pass


def _session_resume_guard(startup_state=None):
    state = startup_state if isinstance(startup_state, dict) else _load_startup_state()
    model_ready = client is not None
    env_file_exists = ENV_FILE.exists()
    if not SESSION_STATE_ENABLED:
        return {
            "allowed": False,
            "reason": "session state is disabled",
            "config_ready": model_ready,
            "model_ready": model_ready,
            "env_file_exists": env_file_exists,
            "startup_state": state,
        }
    if not model_ready:
        return {
            "allowed": False,
            "reason": "local model config is not ready yet",
            "config_ready": model_ready,
            "model_ready": model_ready,
            "env_file_exists": env_file_exists,
            "startup_state": state,
        }
    if not bool(state.get("startup_completed", False)):
        return {
            "allowed": False,
            "reason": "complete one successful startup before auto-resume",
            "config_ready": model_ready,
            "model_ready": model_ready,
            "env_file_exists": env_file_exists,
            "startup_state": state,
        }
    return {
        "allowed": True,
        "reason": "",
        "config_ready": model_ready,
        "model_ready": model_ready,
        "env_file_exists": env_file_exists,
        "startup_state": state,
    }


def _mark_successful_startup(reason: str = "direct_answer", mcp_status=None):
    status = mcp_status or get_mcp_runtime_status()
    existing = _load_startup_state()
    prior_count = int(existing.get("successful_startup_count", 0) or 0)
    normalized_reason = str(reason or "direct_answer").strip() or "direct_answer"
    payload = {
        "startup_completed": True,
        "startup_completion_reason": normalized_reason,
        "startup_used_tools": normalized_reason == "tool_assisted_answer",
        "successful_startup_at": datetime.now().isoformat(timespec="seconds"),
        "successful_startup_count": prior_count + 1,
        "env_file_exists": ENV_FILE.exists(),
        "model_provider": MODEL_PROVIDER,
        "model": MODEL,
        "mcp_connected": bool(status.get("connected", False)),
        "mcp_tool_count": int(status.get("tool_count", 0) or 0),
    }
    _save_startup_state(payload)


def _load_session_state_messages():
    if not SESSION_STATE_ENABLED:
        return []
    try:
        if not SESSION_STATE_FILE.exists():
            return []
        payload = json.loads(SESSION_STATE_FILE.read_text(encoding="utf-8", errors="replace"))
        stored = payload.get("messages", [])
        if not isinstance(stored, list):
            return []
        normalized = []
        for item in stored[-SESSION_STATE_MAX_MESSAGES:]:
            valid = _normalize_saved_chat_message(item)
            if valid:
                normalized.append(valid)
        return normalized
    except Exception:
        return []


def _clear_session_state():
    try:
        if SESSION_STATE_FILE.exists():
            SESSION_STATE_FILE.unlink()
            return True
    except Exception:
        pass
    return False


def _save_session_state(messages, session_trusted: bool = False):
    if not SESSION_STATE_ENABLED:
        return False
    if not session_trusted:
        return False
    try:
        trimmed = []
        for item in messages[-(SESSION_STATE_MAX_MESSAGES * 2):]:
            valid = _normalize_saved_chat_message(item)
            if valid:
                valid["content"] = _redact_for_memory(valid["content"])
                trimmed.append(valid)
        trimmed = trimmed[-SESSION_STATE_MAX_MESSAGES:]
        payload = {
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "model": MODEL,
            "message_count": len(trimmed),
            "messages": trimmed,
        }
        SESSION_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        SESSION_STATE_FILE.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
        return True
    except Exception:
        # Session persistence should never break active chat.
        return False


async def _memory_recall_context(user_prompt: str):
    if not MEMORY_AUTO_RECALL:
        return None

    query = str(user_prompt or "").strip()
    if not query:
        return None

    if AGENT_FAST_MODE:
        lowered = query.lower()
        if len(query) < 12 or len(query.split()) < 3:
            return None
        if lowered in {"ok", "yes", "no", "next", "continue", "done", "thanks"}:
            return None
    else:
        if len(query) < 3:
            return None

    memory_search_fn = AVAILABLE_FUNCTIONS.get("memory_search")
    if not memory_search_fn:
        return None

    try:
        raw = await memory_search_fn(
            {
                "query": query,
                "top_k": MEMORY_RECALL_TOP_K,
                "days_back": MEMORY_RECALL_DAYS_BACK,
                "include_long_term": MEMORY_PRIVATE_SESSION,
                "use_semantic": True,
                "rebuild_index": False,
            }
        )
        parsed = json.loads(raw)
        if parsed.get("status") != "ok":
            return None
        results = parsed.get("results", [])
        if not isinstance(results, list) or not results:
            return None

        lines = []
        for idx, item in enumerate(results[:MEMORY_RECALL_TOP_K], start=1):
            if not isinstance(item, dict):
                continue
            file_path = str(item.get("file", "")).strip()
            snippet = re.sub(r"\s+", " ", str(item.get("snippet", "")).strip())
            if not snippet:
                continue
            if len(snippet) > 260:
                snippet = snippet[:260] + "..."
            lines.append(f"{idx}. {file_path}: {snippet}")

        if not lines:
            return None

        context = "[Memory recall for this turn]\n" + "\n".join(lines)
        if len(context) > MEMORY_RECALL_MAX_CHARS:
            context = context[:MEMORY_RECALL_MAX_CHARS] + "\n...[TRUNCATED]"
        return context
    except Exception:
        return None


async def _execute_tool_call(function_name: str, function_args: dict):
    requested_name = str(function_name or "").strip()
    resolved_name, function_to_call, suggestions = _resolve_tool_callable(requested_name)
    normalized_args = _normalize_tool_arguments(function_args)

    if not function_to_call:
        return json.dumps(
            {
                "status": "failed",
                "tool": requested_name,
                "error": f"Tool '{requested_name}' not found.",
                "suggestions": suggestions,
                "available_tools": len(AVAILABLE_FUNCTIONS),
            },
            ensure_ascii=True,
        )

    try:
        return await asyncio.wait_for(
            execute_tool_with_policy(resolved_name, normalized_args, requested_tool_name=requested_name),
            timeout=TOOL_TIMEOUT_SEC,
        )
    except asyncio.TimeoutError:
        return json.dumps(
            {
                "status": "failed",
                "tool": resolved_name or requested_name,
                "error": f"Tool timeout after {TOOL_TIMEOUT_SEC} seconds.",
            },
            ensure_ascii=True,
        )
    except Exception as e:
        return json.dumps(
            {
                "status": "failed",
                "tool": resolved_name or requested_name,
                "error": str(e),
            },
            ensure_ascii=True,
        )


async def _build_base_messages():
    base_messages = [
        {"role": "system", "content": _load_system_prompt()},
        {"role": "system", "content": RUNTIME_EXECUTION_GUIDE},
    ]
    memory_context = await _memory_bootstrap_context()
    if memory_context:
        files = ", ".join(memory_context.get("files", []))
        base_messages.append(
            {
                "role": "system",
                "content": (
                    f"[Memory bootstrap loaded]\n"
                    f"Files: {files}\n"
                    f"{memory_context.get('content', '')}"
                ),
            }
        )
    return base_messages, memory_context


def _format_runtime_status(messages):
    role_counts = {"system": 0, "user": 0, "assistant": 0, "tool": 0}
    for msg in messages:
        role = str(msg.get("role", "")).strip().lower() if isinstance(msg, dict) else ""
        if role in role_counts:
            role_counts[role] += 1

    mcp_status = get_mcp_runtime_status()
    startup_state = _load_startup_state()
    resume_guard = _session_resume_guard(startup_state=startup_state)
    mcp_browser_tools = sorted(name for name in AVAILABLE_FUNCTIONS if name.startswith("browser_"))
    browser_tools_usable = bool(mcp_status.get("connected") and len(mcp_browser_tools) > 0)
    session_file_exists = SESSION_STATE_FILE.exists()
    saved_at = "n/a"
    saved_count = 0
    if session_file_exists:
        try:
            payload = json.loads(SESSION_STATE_FILE.read_text(encoding="utf-8", errors="replace"))
            saved_at = str(payload.get("saved_at", "n/a"))
            saved_count = int(payload.get("message_count", 0) or 0)
        except Exception:
            saved_at = "unreadable"

    lines = [
        "Agent Runtime Status",
        f"- model: {MODEL}",
        f"- model_provider: {MODEL_PROVIDER}",
        f"- model_ready: {client is not None}",
        f"- model_setup_error: {MODEL_SETUP_ERROR or 'none'}",
        f"- env_file: {ENV_FILE} (exists={ENV_FILE.exists()})",
        f"- env_example_file: {ENV_EXAMPLE_FILE} (exists={ENV_EXAMPLE_FILE.exists()})",
        f"- mcp_status: {mcp_status.get('status', 'unknown')}",
        f"- mcp_connected: {mcp_status.get('connected', False)}",
        f"- mcp_tool_count: {mcp_status.get('tool_count', 0)}",
        f"- mcp_last_error: {mcp_status.get('error', '') or 'none'}",
        f"- browser_tools_registered: {len(mcp_browser_tools)}",
        f"- browser_tools_usable: {browser_tools_usable}",
        f"- max_iterations: {MAX_ITERATIONS}",
        f"- fast_mode: {AGENT_FAST_MODE}",
        f"- tool_timeout_sec: {TOOL_TIMEOUT_SEC}",
        f"- memory_auto_log: {MEMORY_AUTO_LOG}",
        f"- memory_auto_recall: {MEMORY_AUTO_RECALL} (top_k={MEMORY_RECALL_TOP_K}, days_back={MEMORY_RECALL_DAYS_BACK})",
        f"- private_session: {MEMORY_PRIVATE_SESSION}",
        f"- session_state_enabled: {SESSION_STATE_ENABLED}",
        f"- session_state_file: {SESSION_STATE_FILE}",
        f"- startup_state_file: {STARTUP_STATE_FILE}",
        f"- startup_completed: {bool(startup_state.get('startup_completed', False))}",
        f"- startup_completion_reason: {startup_state.get('startup_completion_reason', '') or 'none'}",
        f"- startup_used_tools: {_format_startup_used_tools(startup_state)}",
        f"- startup_trust: {_describe_startup_trust(startup_state)}",
        f"- successful_startup_at: {startup_state.get('successful_startup_at', '') or 'none'}",
        f"- successful_startup_count: {int(startup_state.get('successful_startup_count', 0) or 0)}",
        f"- resume_state: {_describe_resume_state(resume_guard, startup_state)}",
        f"- session_auto_resume_ready: {resume_guard.get('allowed', False)}",
        f"- session_resume_reason: {resume_guard.get('reason', '') or 'none'}",
        f"- session_state_exists: {session_file_exists} (saved_at={saved_at}, message_count={saved_count})",
        f"- loaded_messages: {len(messages)} (system={role_counts['system']}, user={role_counts['user']}, assistant={role_counts['assistant']}, tool={role_counts['tool']})",
        f"- callable_tools: {len(AVAILABLE_FUNCTIONS)}",
    ]
    lines.extend(_format_mcp_proxy_status_lines(mcp_status))
    return "\n".join(lines)


def _help_text():
    return (
        "Commands:\n"
        "- /help: show available commands\n"
        "- /status: show runtime status, memory flags, and session continuity details\n"
        "- /mcp: show direct MCP proxy readiness/trust details when browser tools are connected\n"
        "- /setup or /preflight: show model config, env, and MCP readiness\n"
        "- /doctor: run agent/tool architecture health report\n"
        "- /save: save current conversation state immediately\n"
        "- /reindex: rebuild memory vector index\n"
        "- /reset: clear current chat history and reload base system + memory bootstrap context\n"
        "- Docs: README.md | docs/MODES.md | docs/TROUBLESHOOTING.md\n"
        "- exit or quit: end the agent"
    )


def _trim_messages_for_context(messages):
    if len(messages) <= MAX_HISTORY_MESSAGES:
        return messages
    if not messages:
        return messages

    # Preserve the leading system-message block so core runtime instructions survive long chats.
    leading_system = []
    for item in messages:
        if not isinstance(item, dict):
            break
        if item.get("role") != "system":
            break
        leading_system.append(item)
        if len(leading_system) >= 6:
            break

    if not leading_system:
        leading_system = messages[0:1]

    remaining_slots = max(1, MAX_HISTORY_MESSAGES - len(leading_system))
    tail = messages[-remaining_slots:]
    trimmed = leading_system + tail
    if len(messages) > len(trimmed):
        trimmed.insert(
            len(leading_system),
            {
                "role": "assistant",
                "content": f"[Context trimmed] Retained latest {len(trimmed)-1} messages for performance.",
            },
        )
    return trimmed

async def run_agent():
    """
    Runs the autonomous agent loop in an interactive chat session, supporting async MCP tools.
    """
    base_messages, memory_context = await _build_base_messages()
    messages = list(base_messages)

    if memory_context:
        print("  [Memory] Loaded startup memory context.")

    resume_guard = _session_resume_guard()
    startup_state = resume_guard.get("startup_state") if isinstance(resume_guard, dict) else {}
    resume_summary = _describe_resume_state(
        resume_guard if isinstance(resume_guard, dict) else {},
        startup_state if isinstance(startup_state, dict) else {},
    )
    resumed_messages = _load_session_state_messages() if resume_guard.get("allowed") else []
    startup_marked_for_run = False
    if resumed_messages:
        messages.extend(resumed_messages)
        print(f"  [Session] Loaded {len(resumed_messages)} messages from previous run.")
    elif SESSION_STATE_ENABLED and SESSION_STATE_FILE.exists() and not resume_guard.get("allowed"):
        print(f"  [Session] Auto-resume skipped: {resume_summary}.")
    print(f"  [Session] Resume state: {resume_summary}.")

    print("\n" + "="*50)
    print("Agent is ready! Type '/help' for commands, or 'exit'/'quit' to end the chat.")
    print("="*50 + "\n")

    try:
        while True:
            try:
                user_prompt = input("You: ")
            except (KeyboardInterrupt, EOFError):
                print("\nExiting chat...")
                break

            normalized_input = user_prompt.strip()
            normalized_lower = normalized_input.lower()

            if normalized_lower in ["exit", "quit"]:
                print("Goodbye!")
                break

            if not normalized_input:
                continue

            if normalized_input.startswith("/"):
                if normalized_lower in {"/help", "/?"}:
                    print("\n" + _help_text() + "\n")
                elif normalized_lower == "/status":
                    print("\n" + _format_runtime_status(messages) + "\n")
                elif normalized_lower in {"/mcp", "/proxy"}:
                    print("\n" + _format_mcp_proxy_status_report() + "\n")
                elif normalized_lower in {"/doctor", "/health"}:
                    health_tool = AVAILABLE_FUNCTIONS.get("agent_health_report")
                    if health_tool:
                        try:
                            health_result = await health_tool({"include_tools": False})
                            print(f"\n[Health] {str(health_result)[:1800]}\n")
                        except Exception as e:
                            print(f"\n[Health] Report failed: {e}\n")
                    else:
                        print("\n[Health] agent_health_report tool is not available.\n")
                elif normalized_lower in {"/setup", "/preflight"}:
                    print("\n" + _format_preflight_report() + "\n")
                elif normalized_lower in {"/save", "/checkpoint"}:
                    resume_state = "ready for auto-resume" if startup_marked_for_run else "locked until first successful turn"
                    if _save_session_state(messages, session_trusted=startup_marked_for_run):
                        print("\n[Session] State saved.\n")
                    else:
                        print(f"\n[Session] Checkpoint skipped: resume state is {resume_state}.\n")
                elif normalized_lower == "/reindex":
                    reindex_tool = AVAILABLE_FUNCTIONS.get("memory_reindex")
                    if reindex_tool:
                        try:
                            reindex_result = await reindex_tool({"force_rebuild": True})
                            print(f"\n[Memory] Reindex result: {str(reindex_result)[:600]}\n")
                        except Exception as e:
                            print(f"\n[Memory] Reindex failed: {e}\n")
                    else:
                        print("\n[Memory] memory_reindex tool is not available.\n")
                elif normalized_lower in {"/reset", "/clear"}:
                    base_messages, memory_context = await _build_base_messages()
                    messages = list(base_messages)
                    _clear_session_state()
                    persisted = _save_session_state(messages, session_trusted=startup_marked_for_run)
                    memory_msg = "with memory bootstrap loaded" if memory_context else "without memory bootstrap data"
                    persistence_msg = (
                        "resume state refreshed (ready for auto-resume)"
                        if persisted
                        else "resume state cleared; resume state is locked until first successful turn"
                    )
                    print(f"\n[Session] Conversation reset complete ({memory_msg}; {persistence_msg}).\n")
                else:
                    print("\nUnknown command. Type /help for available commands.\n")
                continue

            # Add the user's new message to the history
            messages.append({"role": "user", "content": normalized_input})
            await _memory_log_event(normalized_input, role="user", importance=3, tags=["conversation"])

            # Preflight local workspace context when user references local paths/codebase.
            preflight_note = await _preflight_local_context(normalized_input)
            if preflight_note:
                messages.append({"role": "assistant", "content": f"[Local workspace preflight]\n{preflight_note}"})
                print("  [Preflight] Local workspace inspection completed.")

            # Dynamic memory recall to improve continuity and relevance each turn.
            recall_note = await _memory_recall_context(normalized_input)
            if recall_note:
                messages.append({"role": "system", "content": recall_note})
                print("  [Memory] Recalled relevant context for this turn.")

            turn_mcp_status = get_mcp_runtime_status()
            turn_tool_context = _build_turn_tool_context(normalized_input, mcp_status=turn_mcp_status)
            turn_tool_context_token = activate_turn_tool_context(turn_tool_context)
            print(f"  [Router] {summarize_turn_tool_context(turn_tool_context)}")

            iteration = 0
            turn_completed = False
            turn_used_tools = False
            turn_tool_results = []
            try:
                while iteration < MAX_ITERATIONS:
                    iteration += 1

                    # Call the OpenAI API
                    try:
                        messages = _trim_messages_for_context(messages)
                        response = client.chat.completions.create(
                            model=MODEL,
                            messages=messages,
                            tools=turn_tool_context.get("tool_schemas", AGENT_TOOLS),
                            tool_choice="auto",  # The model decides whether to call a tool or not
                        )
                    except Exception as e:
                        api_error = f"API Error: {e}"
                        print(api_error)
                        messages.append({"role": "assistant", "content": api_error})
                        turn_completed = True
                        break

                    response_message = response.choices[0].message

                    # Check if the model decided to call a tool
                    tool_calls = response_message.tool_calls

                    if tool_calls:
                        turn_used_tools = True
                        print("  [Agent is thinking... requested to use tool(s)]")
                        # Append the assistant's request to the conversation history
                        messages.append(response_message)

                        # Execute each requested tool
                        for tool_call in tool_calls:
                            function_name = tool_call.function.name
                            function_args = _parse_tool_arguments(tool_call.function.arguments)
                            print(f"  -> Calling '{function_name}' with arguments: {function_args}")

                            function_response = await _execute_tool_call(function_name, function_args)
                            print_resp = str(function_response)
                            print(f"  <- Result from tool: {print_resp[:300]}{'...' if len(print_resp) > 300 else ''}")
                            turn_tool_results.append(function_response)

                            # Append the tool's response to the conversation history
                            messages.append(
                                {
                                    "tool_call_id": tool_call.id,
                                    "role": "tool",
                                    "name": function_name,
                                    "content": str(function_response),
                                }
                            )

                        # Continue loop so the model can consume tool outputs.
                        continue

                    # If there are no tool calls, the agent is finished and provides a standard text response
                    final_answer = response_message.content or ""
                    if (
                        final_answer
                        and _looks_like_local_access_request(normalized_input)
                        and _looks_like_local_access_refusal(final_answer)
                    ):
                        final_answer = await _auto_local_access_fallback(normalized_input)
                    if not final_answer.strip():
                        final_answer = "I completed execution but returned no text summary. Ask me to summarize the results."
                    print(f"\nAgent: {final_answer}\n")
                    await _memory_log_event(
                        final_answer,
                        role="assistant",
                        importance=3,
                        tags=["response"],
                    )

                    # Save the assistant's final response back into the history
                    messages.append({"role": "assistant", "content": final_answer})
                    if not startup_marked_for_run:
                        trust_reason = _trusted_startup_reason_for_turn(turn_used_tools, turn_tool_results)
                        if trust_reason:
                            _mark_successful_startup(reason=trust_reason)
                            startup_marked_for_run = True
                        elif turn_used_tools:
                            print("  [Session] Resume state remains locked: tool-assisted turn did not finish with successful tool results.")
                    turn_completed = True
                    break

                if iteration == MAX_ITERATIONS and not turn_completed:
                    timeout_msg = "Agent Error: Reached maximum internal steps without producing a final answer."
                    print(f"\n{timeout_msg} Continuing to next turn...\n")
                    messages.append({"role": "assistant", "content": timeout_msg})
            finally:
                reset_turn_tool_context(turn_tool_context_token)
    finally:
        _save_session_state(messages, session_trusted=startup_marked_for_run)

async def main():
    if _wants_preflight_mode():
        mcp_status = await init_mcp_client()
        try:
            print(_format_preflight_report(mcp_status))
        finally:
            await shutdown_mcp_client()
        return

    if _wants_mcp_status_mode():
        previous_quiet = os.environ.get("AGENT_MCP_QUIET")
        if _wants_json_output():
            os.environ["AGENT_MCP_QUIET"] = "1"
        mcp_status = await init_mcp_client()
        try:
            if _wants_json_output():
                print(json.dumps(_build_mcp_proxy_status_payload(mcp_status), ensure_ascii=True, indent=2))
            else:
                print(_format_mcp_proxy_status_report(mcp_status))
        finally:
            await shutdown_mcp_client()
            if previous_quiet is None:
                os.environ.pop("AGENT_MCP_QUIET", None)
            else:
                os.environ["AGENT_MCP_QUIET"] = previous_quiet
        return

    if client is not None:
        print("\n" + _format_startup_banner())
        try:
            # Initialize the MCP Client and connection before running
            mcp_status = await init_mcp_client()
            print(_format_mcp_startup_note(mcp_status or get_mcp_runtime_status()))
            await run_agent()
        finally:
            await shutdown_mcp_client()
    else:
        print("Cannot run Python agent: model client is not configured.")
        print(f"Requested provider: {MODEL_PROVIDER}")
        print(f"Model setup error: {MODEL_SETUP_ERROR or 'unknown configuration error'}")
        print(_format_model_setup_help())
        print("\n" + _format_preflight_report() + "\n")

if __name__ == "__main__":
    asyncio.run(main())
    
