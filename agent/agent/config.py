"""
Agent configuration: environment loading, model client creation, and runtime constants.

Extracted from agent.py to keep the conversation loop focused on execution.
"""
import os
from pathlib import Path
from dotenv import load_dotenv
from openai import AzureOpenAI, OpenAI

load_dotenv()


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name, "true" if default else "false")
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except Exception:
        return default


# --- Agent behavior constants ------------------------------------------------

MAX_ITERATIONS = max(2, min(_env_int("AGENT_MAX_ITERATIONS", 10), 40))
MAX_HISTORY_MESSAGES = max(20, min(_env_int("AGENT_MAX_HISTORY_MESSAGES", 60), 200))
TOOL_TIMEOUT_SEC = float(os.environ.get("AGENT_TOOL_TIMEOUT_SEC", "180"))
MEMORY_AUTO_LOG = _env_flag("AGENT_MEMORY_AUTO_LOG", False)
MEMORY_PRIVATE_SESSION = _env_flag("AGENT_PRIVATE_SESSION", True)
MAX_MEMORY_LOG_CHARS = max(600, min(_env_int("AGENT_MEMORY_LOG_MAX_CHARS", 4000), 50000))
MEMORY_AUTO_RECALL = _env_flag("AGENT_MEMORY_AUTO_RECALL", True)
MEMORY_RECALL_TOP_K = max(1, min(_env_int("AGENT_MEMORY_RECALL_TOP_K", 4), 12))
MEMORY_RECALL_DAYS_BACK = max(1, min(_env_int("AGENT_MEMORY_RECALL_DAYS_BACK", 30), 180))
MEMORY_RECALL_MAX_CHARS = max(600, min(_env_int("AGENT_MEMORY_RECALL_MAX_CHARS", 3500), 30000))
SESSION_STATE_ENABLED = _env_flag("AGENT_SESSION_STATE_ENABLED", True)
SESSION_STATE_MAX_MESSAGES = max(8, min(_env_int("AGENT_SESSION_STATE_MAX_MESSAGES", 80), 300))
SESSION_STATE_MAX_TURNS = max(1, min(_env_int("AGENT_SESSION_STATE_MAX_TURNS", 12), 50))
TURN_GROUNDING_MAX_CHARS = max(800, min(_env_int("AGENT_TURN_GROUNDING_MAX_CHARS", 4000), 12000))
SESSION_STATE_FILE = Path(
    os.environ.get(
        "AGENT_SESSION_STATE_FILE",
        str(Path(__file__).resolve().parents[2] / ".agent-state" / "last_session.json"),
    )
)

RUNTIME_EXECUTION_GUIDE = (
    "[Runtime execution guide]\n"
    "- For broad goals, prefer task_autopilot first to map files and next actions.\n"
    "- For deterministic multi-step execution, use workflow_execute with explicit per-step verification.\n"
    "- For edits, prefer fs tools and verify with fs_read/fs_search before final response.\n"
    "- When tool output is ambiguous, call the tool_catalog or a validation tool instead of guessing.\n"
    "- Final responses must stay matched to verified tool results, saved session state, or memory recall; if something is not verified, say so explicitly."
)


# --- Model client creation ---------------------------------------------------

def create_client_and_model():
    """
    Returns (client, model_name, provider, setup_error).
    client is None when credentials are missing.
    """
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


# --- Module-level singleton (backward compatible) ----------------------------

client, MODEL, MODEL_PROVIDER, MODEL_SETUP_ERROR = create_client_and_model()
