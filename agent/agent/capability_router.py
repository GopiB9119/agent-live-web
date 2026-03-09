from contextvars import ContextVar
from typing import Any, Iterable


TURN_TOOL_CONTEXT: ContextVar[dict | None] = ContextVar("turn_tool_context", default=None)

CORE_TOOL_NAMES = {"calculate", "reasoning_plan"}
DIAGNOSTIC_TOOL_NAMES = {"tool_catalog", "agent_health_report", "agent_proxy_status"}
WORKFLOW_TOOL_NAMES = {"task_autopilot", "workflow_execute"}
MEMORY_TOOL_NAMES = {"memory_search", "memory_get"}
WEB_TOOL_NAMES = {"web_fetch", "oauth_get_token", "oauth_profiles", "oauth_set_profile"}
WORKSPACE_READ_TOOL_NAMES = {
    "fs_list",
    "fs_read",
    "fs_read_batch",
    "fs_search",
    "fs_analyze_file",
    "codebase_analyze",
}
WORKSPACE_WRITE_TOOL_NAMES = {
    "fs_edit_lines",
    "fs_insert_lines",
    "fs_write",
    "fs_copy",
    "fs_move",
    "fs_patch",
}
WORKSPACE_DELETE_TOOL_NAMES = {"fs_delete"}
WORKSPACE_COMMAND_TOOL_NAMES = {"run_command"}
LOCAL_BROWSER_HELPER_TOOL_NAMES = {
    "browser_tabs_list",
    "browser_tab_select",
    "browser_close_blank_tabs",
}
BROWSER_READONLY_TOOL_NAMES = {
    "browser_tabs",
    "browser_snapshot",
    "browser_take_screenshot",
    "browser_console_messages",
    "browser_network_requests",
}
BROWSER_NAVIGATION_TOOL_NAMES = {
    "browser_navigate",
    "browser_navigate_back",
    "browser_wait_for",
    "browser_hover",
    "browser_resize",
    "browser_press_key",
}
BROWSER_MUTATING_TOOL_NAMES = {
    "browser_click",
    "browser_type",
    "browser_fill_form",
    "browser_select_option",
    "browser_drag",
    "browser_mouse_click_xy",
    "browser_mouse_drag_xy",
    "browser_mouse_move_xy",
    "browser_handle_dialog",
    "browser_file_upload",
}
EXCLUDED_BY_DEFAULT_TOOL_NAMES = {"call_tool"}


def _clean_name(value: Any) -> str:
    return str(value or "").strip()


def _schema_name(item: Any) -> str:
    if not isinstance(item, dict):
        return ""
    function = item.get("function", {})
    if not isinstance(function, dict):
        return ""
    return _clean_name(function.get("name"))


def _normalize_name_set(names: Iterable[Any]) -> set[str]:
    normalized = set()
    for item in names:
        clean = _clean_name(item)
        if clean:
            normalized.add(clean)
    return normalized


def _wants_command_tools(task_spec: dict) -> bool:
    mode = _clean_name(task_spec.get("task_mode")).lower()
    task_text = _clean_name(task_spec.get("task")).lower()
    requested_actions = {item.lower() for item in task_spec.get("requested_actions", []) if isinstance(item, str)}
    return mode in {"debug", "test"} or any(token in task_text for token in {"terminal", "command", "npm", "pytest", "pip", "lint", "typecheck", "build"}) or "test" in requested_actions


def _wants_memory_tools(task_spec: dict) -> bool:
    task_text = _clean_name(task_spec.get("task")).lower()
    return any(token in task_text for token in {"remember", "recall", "history", "previous context"})


def _wants_web_tools(task_spec: dict) -> bool:
    task_text = _clean_name(task_spec.get("task")).lower()
    actions = {item.lower() for item in task_spec.get("requested_actions", []) if isinstance(item, str)}
    return "fetch" in actions or any(token in task_text for token in {"api", "oauth", "token", "http request", "web fetch"})


def _resolve_surfaces(task_spec: dict, available_names: set[str], mcp_status: dict | None = None) -> tuple[list[str], list[str]]:
    surfaces = ["core"]
    reasons = list(task_spec.get("reason_codes", []) or [])
    primary_kind = _clean_name(task_spec.get("primary_kind")).lower()
    task_mode = _clean_name(task_spec.get("task_mode")).lower()
    hints = {item.lower() for item in task_spec.get("surfaces_hint", []) if isinstance(item, str)}
    browser_available = bool(
        (mcp_status or {}).get("connected")
        and any(name.startswith("browser_") for name in available_names)
    )

    if primary_kind in {"repo", "hybrid", "diagnostics", "python"} or "workspace" in hints or task_mode in {"draft", "mutate", "debug", "test", "submit"}:
        surfaces.append("workspace")
    if primary_kind in {"browser", "hybrid"} or "browser" in hints:
        if browser_available:
            surfaces.append("browser")
        else:
            reasons.append("browser_surface_requested_but_unavailable")
    if primary_kind == "python" or "python" in hints:
        surfaces.append("python")
    if primary_kind in {"diagnostics", "python"} or "diagnostics" in hints or task_mode in {"debug", "test"}:
        surfaces.append("diagnostics")
    if primary_kind == "platform" or "github" in hints:
        surfaces.append("github")
    if _wants_web_tools(task_spec):
        surfaces.append("web")
    if _wants_memory_tools(task_spec):
        surfaces.append("memory")
    if any(surface in surfaces for surface in {"workspace", "browser"}):
        surfaces.append("workflow")

    ordered = []
    for surface in surfaces:
        if surface not in ordered:
            ordered.append(surface)
    return ordered, reasons


def _browser_tool_names_for_task(task_spec: dict, available_names: set[str]) -> set[str]:
    task_mode = _clean_name(task_spec.get("task_mode")).lower()
    requested_actions = {item.lower() for item in task_spec.get("requested_actions", []) if isinstance(item, str)}
    allowed = set(LOCAL_BROWSER_HELPER_TOOL_NAMES)
    allowed.update(BROWSER_READONLY_TOOL_NAMES)
    allowed.update(BROWSER_NAVIGATION_TOOL_NAMES)

    if task_mode in {"test", "mutate", "submit", "debug"}:
        allowed.update(BROWSER_MUTATING_TOOL_NAMES)

    if "upload" in requested_actions:
        allowed.add("browser_file_upload")
    if "download" in requested_actions:
        allowed.add("browser_pdf_save")

    return allowed & available_names


def _workspace_tool_names_for_task(task_spec: dict, available_names: set[str]) -> set[str]:
    task_mode = _clean_name(task_spec.get("task_mode")).lower()
    requested_actions = {item.lower() for item in task_spec.get("requested_actions", []) if isinstance(item, str)}
    allowed = set(WORKSPACE_READ_TOOL_NAMES)

    if task_mode in {"draft", "mutate", "submit", "debug"} or any(action in requested_actions for action in {"edit", "create"}):
        allowed.update(WORKSPACE_WRITE_TOOL_NAMES)

    if "delete" in requested_actions:
        allowed.update(WORKSPACE_DELETE_TOOL_NAMES)

    if _wants_command_tools(task_spec):
        allowed.update(WORKSPACE_COMMAND_TOOL_NAMES)

    return allowed & available_names


def _python_tool_names_for_task(available_names: set[str]) -> set[str]:
    allowed = set()
    for name in available_names:
        lowered = name.lower()
        if lowered.startswith("pylance") or lowered.startswith("python_") or "python_environment" in lowered:
            allowed.add(name)
    return allowed


def _github_tool_names_for_task(task_spec: dict, available_names: set[str]) -> set[str]:
    task_mode = _clean_name(task_spec.get("task_mode")).lower()
    allowed = set()
    for name in available_names:
        if not name.lower().startswith("github_"):
            continue
        lowered = name.lower()
        if any(token in lowered for token in {"merge", "push", "delete", "create_repository"}):
            continue
        if task_mode in {"inspect", "debug", "test"} and any(token in lowered for token in {"get_", "list_", "search_", "issue_read", "pull_request_read"}):
            allowed.add(name)
        elif task_mode in {"draft", "mutate", "submit"} and any(token in lowered for token in {"get_", "list_", "search_", "add_comment", "create_branch", "update_pull_request"}):
            allowed.add(name)
    return allowed


def _build_allowed_tool_names(task_spec: dict, available_names: set[str], surfaces: list[str]) -> set[str]:
    allowed = set(name for name in CORE_TOOL_NAMES if name in available_names)
    task_mode = _clean_name(task_spec.get("task_mode")).lower()

    if "diagnostics" in surfaces:
        allowed.update(name for name in DIAGNOSTIC_TOOL_NAMES if name in available_names)
    if "workspace" in surfaces:
        allowed.update(_workspace_tool_names_for_task(task_spec, available_names))
    if "browser" in surfaces:
        allowed.update(_browser_tool_names_for_task(task_spec, available_names))
    if "python" in surfaces:
        allowed.update(_python_tool_names_for_task(available_names))
    if "github" in surfaces:
        allowed.update(_github_tool_names_for_task(task_spec, available_names))
    if "web" in surfaces:
        allowed.update(name for name in WEB_TOOL_NAMES if name in available_names)
    if "memory" in surfaces:
        allowed.update(name for name in MEMORY_TOOL_NAMES if name in available_names)
    if "workflow" in surfaces and "task_autopilot" in available_names:
        allowed.add("task_autopilot")
        if task_mode in {"test", "mutate", "submit", "debug"} and "workflow_execute" in available_names:
            allowed.add("workflow_execute")

    allowed.difference_update(EXCLUDED_BY_DEFAULT_TOOL_NAMES)
    return allowed


def summarize_turn_tool_context(context: dict) -> str:
    task_spec = context.get("task_spec", {}) if isinstance(context, dict) else {}
    task_mode = _clean_name(task_spec.get("task_mode")).lower() or "unknown"
    primary_kind = _clean_name(task_spec.get("primary_kind")).lower() or "unknown"
    surfaces = ",".join(context.get("surfaces", []) or []) or "none"
    tool_count = int(context.get("tool_count", 0) or 0)
    risk_level = _clean_name(task_spec.get("risk_level")).lower() or "unknown"
    return f"mode={task_mode} kind={primary_kind} risk={risk_level} surfaces={surfaces} tools={tool_count}"


def build_turn_tool_context(task_spec: dict, agent_tools: list[dict], available_tool_names: Iterable[Any], mcp_status: dict | None = None) -> dict:
    available_names = _normalize_name_set(available_tool_names)
    surfaces, reasons = _resolve_surfaces(task_spec, available_names, mcp_status=mcp_status)
    allowed_tool_names = _build_allowed_tool_names(task_spec, available_names, surfaces)
    filtered_schemas = []
    allowed_schema_names = []
    for item in agent_tools:
        name = _schema_name(item)
        if name and name in allowed_tool_names:
            filtered_schemas.append(item)
            allowed_schema_names.append(name)

    if not filtered_schemas:
        for item in agent_tools:
            if _schema_name(item) == "calculate":
                filtered_schemas.append(item)
                allowed_schema_names.append("calculate")
                break

    return {
        "task_spec": task_spec,
        "surfaces": surfaces,
        "reason_codes": reasons,
        "allowed_tool_names": sorted(set(allowed_schema_names)),
        "tool_schemas": filtered_schemas,
        "tool_count": len(filtered_schemas),
        "summary": summarize_turn_tool_context(
            {
                "task_spec": task_spec,
                "surfaces": surfaces,
                "tool_count": len(filtered_schemas),
            }
        ),
    }


def activate_turn_tool_context(context: dict):
    return TURN_TOOL_CONTEXT.set(context if isinstance(context, dict) else {})


def reset_turn_tool_context(token) -> None:
    TURN_TOOL_CONTEXT.reset(token)


def get_turn_tool_context() -> dict:
    current = TURN_TOOL_CONTEXT.get()
    return current if isinstance(current, dict) else {}
