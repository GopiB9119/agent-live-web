import json
import inspect
try:
    from oauth_tools import OAuthManager
except Exception:
    from .oauth_tools import OAuthManager
try:
    from memory_tools import MemoryManager
except Exception:
    from .memory_tools import MemoryManager
try:
    from web_tools import WebManager
except Exception:
    from .web_tools import WebManager
try:
    from mcp_tools import MCPManager
except Exception:
    from .mcp_tools import MCPManager
try:
    from fs_tools import FSManager
except Exception:
    from .fs_tools import FSManager
try:
    from command_tools import CommandManager
except Exception:
    from .command_tools import CommandManager
try:
    from workflow_tools import WorkflowManager
except Exception:
    from .workflow_tools import WorkflowManager
try:
    from diagnostics_tools import DiagnosticsManager
except Exception:
    from .diagnostics_tools import DiagnosticsManager
try:
    from architecture.safety_policy import evaluate_tool_call
except Exception:
    from .architecture.safety_policy import evaluate_tool_call
try:
    from architecture.safety_audit import write_safety_event
except Exception:
    from .architecture.safety_audit import write_safety_event
try:
    from capability_router import get_turn_tool_context
except Exception:
    from .capability_router import get_turn_tool_context
try:
    from tooling.registry import (
        auto_register_missing_local_tool_schemas,
        build_base_available_functions,
        build_local_callable_registry,
        ensure_safety_confirm_schema_fields,
        register_or_update_tool_schema,
    )
except Exception:
    from .tooling.registry import (
        auto_register_missing_local_tool_schemas,
        build_base_available_functions,
        build_local_callable_registry,
        ensure_safety_confirm_schema_fields,
        register_or_update_tool_schema,
    )
try:
    from tooling.schemas import AGENT_TOOLS
except Exception:
    from .tooling.schemas import AGENT_TOOLS
try:
    from runtime_utils import (
        BINARY_SUFFIXES,
        LONG_TERM_MEMORY_FILE,
        MEMORY_DIR,
        MEMORY_VECTOR_DIM,
        MEMORY_VECTOR_INDEX_FILE,
        NOISE_DIR_NAMES,
        RUN_COMMAND_ALLOW_DANGEROUS_ENV,
        RUN_COMMAND_SECURITY_MODE_DEFAULT,
        WEB_FETCH_ALLOW_PRIVATE_ENV,
        WORKSPACE_ROOT,
        is_private_or_local_host as _is_private_or_local_host,
        redact_sensitive_text as _redact_sensitive_text,
        resolve_workspace_path as _resolve_workspace_path,
        run_command_is_safe_in_restricted_mode as _run_command_is_safe_in_restricted_mode,
        to_bool as _to_bool,
    )
except Exception:
    from .runtime_utils import (
        BINARY_SUFFIXES,
        LONG_TERM_MEMORY_FILE,
        MEMORY_DIR,
        MEMORY_VECTOR_DIM,
        MEMORY_VECTOR_INDEX_FILE,
        NOISE_DIR_NAMES,
        RUN_COMMAND_ALLOW_DANGEROUS_ENV,
        RUN_COMMAND_SECURITY_MODE_DEFAULT,
        WEB_FETCH_ALLOW_PRIVATE_ENV,
        WORKSPACE_ROOT,
        is_private_or_local_host as _is_private_or_local_host,
        redact_sensitive_text as _redact_sensitive_text,
        resolve_workspace_path as _resolve_workspace_path,
        run_command_is_safe_in_restricted_mode as _run_command_is_safe_in_restricted_mode,
        to_bool as _to_bool,
    )

RETRYABLE_TOOLS = {
    "browser_navigate",
    "browser_click",
    "browser_type",
    "browser_fill_form",
    "browser_select_option",
    "browser_press_key",
}
STATE_CHANGE_TOOLS = {
    "browser_click",
    "browser_type",
    "browser_fill_form",
    "browser_select_option",
    "browser_press_key",
}
OWNERSHIP_SKIP_TOOLS = {"browser_tabs", "browser_close", "browser_install"}

OAUTH_MANAGER = OAuthManager(
    to_bool_fn=_to_bool,
    is_private_or_local_host_fn=_is_private_or_local_host,
    web_fetch_allow_private_env=WEB_FETCH_ALLOW_PRIVATE_ENV,
)
MEMORY_MANAGER = MemoryManager(
    workspace_root=WORKSPACE_ROOT,
    memory_dir=MEMORY_DIR,
    long_term_memory_file=LONG_TERM_MEMORY_FILE,
    vector_index_file=MEMORY_VECTOR_INDEX_FILE,
    vector_dim=MEMORY_VECTOR_DIM,
    resolve_workspace_path_fn=_resolve_workspace_path,
    redact_sensitive_text_fn=_redact_sensitive_text,
)
WEB_MANAGER = WebManager(
    to_bool_fn=_to_bool,
    is_private_or_local_host_fn=_is_private_or_local_host,
    oauth_manager=OAUTH_MANAGER,
    web_fetch_allow_private_env=WEB_FETCH_ALLOW_PRIVATE_ENV,
)
MCP_MANAGER = MCPManager(
    workspace_root=WORKSPACE_ROOT,
    retryable_tools=RETRYABLE_TOOLS,
    state_change_tools=STATE_CHANGE_TOOLS,
    ownership_skip_tools=OWNERSHIP_SKIP_TOOLS,
)
FS_MANAGER = FSManager(
    workspace_root=WORKSPACE_ROOT,
    resolve_workspace_path_fn=_resolve_workspace_path,
    noise_dir_names=NOISE_DIR_NAMES,
    binary_suffixes=BINARY_SUFFIXES,
)
COMMAND_MANAGER = CommandManager(
    workspace_root=WORKSPACE_ROOT,
    resolve_workspace_path_fn=_resolve_workspace_path,
    run_command_security_mode_default=RUN_COMMAND_SECURITY_MODE_DEFAULT,
    run_command_allow_dangerous_env=RUN_COMMAND_ALLOW_DANGEROUS_ENV,
    to_bool_fn=_to_bool,
    run_command_is_safe_in_restricted_mode_fn=_run_command_is_safe_in_restricted_mode,
)
WORKFLOW_MANAGER = WorkflowManager(
    available_functions_provider=lambda: AVAILABLE_FUNCTIONS,
    is_probably_text_source_fn=FS_MANAGER.is_probably_text_source,
    codebase_analyze_fn=FS_MANAGER.codebase_analyze,
    fs_analyze_file_fn=FS_MANAGER.fs_analyze_file,
    tool_invoker_fn=lambda tool_name, arguments: execute_tool_with_policy(tool_name, arguments),
)
DIAGNOSTICS_MANAGER = DiagnosticsManager(
    agent_tools_provider=lambda: AGENT_TOOLS,
    available_functions_provider=lambda: AVAILABLE_FUNCTIONS,
    resolve_workspace_path_fn=_resolve_workspace_path,
    to_bool_fn=_to_bool,
)


# Define the local calculator backup tool
def calculate(expression: str) -> str:
    """Evaluates a basic math expression securely."""
    try:
        allowed_chars = set("0123456789+-*/(). ")
        if not all(c in allowed_chars for c in expression):
            return "Error: Invalid characters in math expression. Only digits and +-*/() are allowed."
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"Error evaluating expression: {str(e)}"


# Define the tools available to the OpenAI model (Starting with the local ones)
AVAILABLE_FUNCTIONS = build_base_available_functions(calculate)


def _summarize_for_audit(value, max_chars: int = 1200):
    text = _redact_sensitive_text(str(value or ""))
    if len(text) > max_chars:
        return text[:max_chars] + "...[TRUNCATED]"
    return text


def _coerce_preview_result(raw_value):
    if raw_value is None:
        return None
    if isinstance(raw_value, dict):
        return raw_value
    try:
        parsed = json.loads(str(raw_value))
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    return {"status": "ok", "preview": {"raw": str(raw_value)}}


async def _build_preview_payload(tool_name: str, arguments: dict, target=None):
    if tool_name == "fs_write":
        return _coerce_preview_result(await FS_MANAGER.preview_fs_write(arguments))
    if tool_name == "fs_copy":
        return _coerce_preview_result(await FS_MANAGER.preview_fs_copy(arguments, move=False))
    if tool_name == "fs_move":
        return _coerce_preview_result(await FS_MANAGER.preview_fs_copy(arguments, move=True))
    if tool_name == "fs_delete":
        return _coerce_preview_result(await FS_MANAGER.preview_fs_delete(arguments))
    if tool_name == "run_command":
        return _coerce_preview_result(await COMMAND_MANAGER.preview_command(arguments))
    if tool_name in {"fs_edit_lines", "fs_insert_lines", "fs_patch"} and callable(target):
        preview_args = dict(arguments or {})
        preview_args["dry_run"] = True
        preview_result = await target(preview_args) if inspect.iscoroutinefunction(target) else target(preview_args)
        return _coerce_preview_result(preview_result)
    return None


def _audit_decision(tool_name: str, requested_tool_name: str, arguments: dict, safety, preview_payload=None):
    if safety.action_class == "read_only" and safety.decision == "allow":
        return
    write_safety_event(
        WORKSPACE_ROOT,
        {
            "event_type": "decision",
            "tool": tool_name,
            "requested_tool": requested_tool_name,
            "action_class": safety.action_class,
            "risk_level": safety.risk_level,
            "decision": safety.decision,
            "reason_codes": safety.reason_codes,
            "arguments_summary": _summarize_for_audit(arguments),
            "preview_attached": bool(preview_payload),
            "confirm_token_issued": bool(getattr(safety, "confirm_token", "")),
        },
        redact_fn=_redact_sensitive_text,
    )


def _audit_execution(tool_name: str, requested_tool_name: str, arguments: dict, safety, result_text: str, status: str, error: str = ""):
    if safety.action_class == "read_only" and safety.decision == "allow":
        return
    write_safety_event(
        WORKSPACE_ROOT,
        {
            "event_type": "execution",
            "tool": tool_name,
            "requested_tool": requested_tool_name,
            "action_class": safety.action_class,
            "risk_level": safety.risk_level,
            "decision": safety.decision,
            "execution_status": status,
            "arguments_summary": _summarize_for_audit(arguments),
            "result_summary": _summarize_for_audit(result_text),
            "error": error,
        },
        redact_fn=_redact_sensitive_text,
    )


async def execute_tool_with_policy(tool_name: str, arguments: dict | None = None, requested_tool_name: str | None = None):
    args = arguments if isinstance(arguments, dict) else {}
    clean_name = str(tool_name or "").strip()
    requested = str(requested_tool_name or clean_name).strip() or clean_name
    turn_context = get_turn_tool_context()
    allowed_tool_names = set(turn_context.get("allowed_tool_names", []) or [])

    if allowed_tool_names and clean_name not in allowed_tool_names:
        payload = {
            "status": "blocked",
            "tool": clean_name,
            "requested_tool": requested,
            "error": f"Tool '{clean_name}' is not exposed for the current task.",
            "reason_code": "tool_not_exposed_for_turn",
            "task_mode": turn_context.get("task_spec", {}).get("task_mode", "unknown"),
            "surfaces": turn_context.get("surfaces", []),
            "available_turn_tools": len(allowed_tool_names),
        }
        write_safety_event(
            WORKSPACE_ROOT,
            {
                "event_type": "routing_block",
                "tool": clean_name,
                "requested_tool": requested,
                "decision": "blocked",
                "reason_codes": ["tool_not_exposed_for_turn"],
                "arguments_summary": _summarize_for_audit(args),
                "task_mode": turn_context.get("task_spec", {}).get("task_mode", "unknown"),
                "surfaces": turn_context.get("surfaces", []),
            },
            redact_fn=_redact_sensitive_text,
        )
        return json.dumps(payload, ensure_ascii=True)

    if clean_name == "call_tool":
        nested_name = str(args.get("tool_name", "")).strip()
        nested_arguments = args.get("arguments", {}) or {}
        if not isinstance(nested_arguments, dict):
            return json.dumps({"status": "failed", "error": "arguments must be an object"}, ensure_ascii=True)
        if not nested_name:
            return json.dumps({"status": "failed", "error": "tool_name is required"}, ensure_ascii=True)
        if nested_name == "call_tool":
            return json.dumps({"status": "failed", "error": "Recursive call_tool is not allowed"}, ensure_ascii=True)
        return await execute_tool_with_policy(nested_name, nested_arguments, requested_tool_name=requested)

    target = AVAILABLE_FUNCTIONS.get(clean_name)
    if not target:
        return json.dumps({"status": "failed", "error": f"Tool not found: {clean_name}"}, ensure_ascii=True)

    safety = evaluate_tool_call(clean_name, args, WORKSPACE_ROOT)
    preview_payload = None
    if safety.decision not in {"allow", "allow_with_verification"}:
        preview_payload = await _build_preview_payload(clean_name, args, target=target)
        payload = {
            "status": safety.decision,
            **safety.to_dict(),
        }
        if preview_payload is not None:
            payload["preview"] = preview_payload
        if requested and requested != clean_name:
            payload["requested_tool"] = requested
            payload["tool_resolution"] = "fuzzy-match"
        _audit_decision(clean_name, requested, args, safety, preview_payload=preview_payload)
        return json.dumps(payload, ensure_ascii=True)

    _audit_decision(clean_name, requested, args, safety)
    try:
        if inspect.iscoroutinefunction(target):
            result = await target(args)
        else:
            try:
                result = target(**args)
            except TypeError:
                result = target(args)
    except Exception as e:
        _audit_execution(clean_name, requested, args, safety, "", "failed", error=str(e))
        return json.dumps({"status": "failed", "tool": clean_name, "error": str(e)}, ensure_ascii=True)

    if requested and requested != clean_name:
        try:
            parsed = json.loads(str(result))
            if isinstance(parsed, dict):
                parsed.setdefault("requested_tool", requested)
                parsed.setdefault("tool_resolution", "fuzzy-match")
                _audit_execution(clean_name, requested, args, safety, json.dumps(parsed, ensure_ascii=True), parsed.get("status", "ok"))
                return json.dumps(parsed, ensure_ascii=True)
        except Exception:
            pass
        _audit_execution(clean_name, requested, args, safety, str(result), "ok")
        return json.dumps(
            {
                "status": "ok",
                "tool": clean_name,
                "requested_tool": requested,
                "tool_resolution": "fuzzy-match",
                "result": str(result),
            },
            ensure_ascii=True,
        )
    _audit_execution(clean_name, requested, args, safety, str(result), "ok")
    return result

async def browser_tabs_list(_kwargs_dict=None):
    return await MCP_MANAGER.browser_tabs_list(_kwargs_dict)


async def browser_tab_select(kwargs_dict):
    return await MCP_MANAGER.browser_tab_select(kwargs_dict)


async def browser_close_blank_tabs(_kwargs_dict=None):
    return await MCP_MANAGER.browser_close_blank_tabs(_kwargs_dict)


async def fs_list(kwargs_dict=None):
    return await FS_MANAGER.fs_list(kwargs_dict)


async def fs_read(kwargs_dict):
    return await FS_MANAGER.fs_read(kwargs_dict)


async def fs_read_batch(kwargs_dict):
    return await FS_MANAGER.fs_read_batch(kwargs_dict)


async def fs_edit_lines(kwargs_dict):
    return await FS_MANAGER.fs_edit_lines(kwargs_dict)


async def fs_insert_lines(kwargs_dict):
    return await FS_MANAGER.fs_insert_lines(kwargs_dict)


async def fs_write(kwargs_dict):
    return await FS_MANAGER.fs_write(kwargs_dict)


async def fs_copy(kwargs_dict):
    return await FS_MANAGER.fs_copy(kwargs_dict)


async def fs_move(kwargs_dict):
    return await FS_MANAGER.fs_move(kwargs_dict)


async def fs_delete(kwargs_dict):
    return await FS_MANAGER.fs_delete(kwargs_dict)


async def fs_patch(kwargs_dict):
    return await FS_MANAGER.fs_patch(kwargs_dict)


async def fs_search(kwargs_dict):
    return await FS_MANAGER.fs_search(kwargs_dict)


async def fs_analyze_file(kwargs_dict):
    return await FS_MANAGER.fs_analyze_file(kwargs_dict)


async def codebase_analyze(kwargs_dict=None):
    return await FS_MANAGER.codebase_analyze(kwargs_dict)


async def reasoning_plan(kwargs_dict):
    return await WORKFLOW_MANAGER.reasoning_plan(kwargs_dict)


async def memory_log(kwargs_dict):
    return await MEMORY_MANAGER.memory_log(kwargs_dict)


async def memory_promote(kwargs_dict):
    return await MEMORY_MANAGER.memory_promote(kwargs_dict)


async def memory_get(kwargs_dict=None):
    return await MEMORY_MANAGER.memory_get(kwargs_dict)


async def memory_search(kwargs_dict):
    return await MEMORY_MANAGER.memory_search(kwargs_dict)


async def memory_bootstrap(kwargs_dict=None):
    return await MEMORY_MANAGER.memory_bootstrap(kwargs_dict)


async def memory_reindex(kwargs_dict=None):
    return await MEMORY_MANAGER.memory_reindex(kwargs_dict)


async def tool_catalog(kwargs_dict=None):
    return await DIAGNOSTICS_MANAGER.tool_catalog(kwargs_dict)


async def agent_health_report(kwargs_dict=None):
    return await DIAGNOSTICS_MANAGER.agent_health_report(kwargs_dict)


async def workflow_execute(kwargs_dict):
    return await WORKFLOW_MANAGER.workflow_execute(kwargs_dict)


async def task_autopilot(kwargs_dict):
    return await WORKFLOW_MANAGER.task_autopilot(kwargs_dict)


async def oauth_set_profile(kwargs_dict):
    return await OAUTH_MANAGER.oauth_set_profile(kwargs_dict)


async def oauth_get_token(kwargs_dict):
    return await OAUTH_MANAGER.oauth_get_token(kwargs_dict)


async def oauth_profiles(kwargs_dict=None):
    return await OAUTH_MANAGER.oauth_profiles(kwargs_dict)


async def run_command(kwargs_dict):
    return await COMMAND_MANAGER.run_command(kwargs_dict)


async def web_fetch(kwargs_dict):
    return await WEB_MANAGER.web_fetch(kwargs_dict)


async def call_tool(kwargs_dict):
    kwargs = kwargs_dict or {}
    tool_name = str(kwargs.get("tool_name", "")).strip()
    arguments = kwargs.get("arguments", {}) or {}
    if not isinstance(arguments, dict):
        return json.dumps({"status": "failed", "error": "arguments must be an object"}, ensure_ascii=True)
    if not tool_name:
        return json.dumps({"status": "failed", "error": "tool_name is required"}, ensure_ascii=True)
    if tool_name == "call_tool":
        return json.dumps({"status": "failed", "error": "Recursive call_tool is not allowed"}, ensure_ascii=True)
    return await execute_tool_with_policy(tool_name, arguments, requested_tool_name="call_tool")


AVAILABLE_FUNCTIONS.update(build_local_callable_registry(globals()))
ensure_safety_confirm_schema_fields(AGENT_TOOLS)

_added_local_schemas = auto_register_missing_local_tool_schemas(
    agent_tools=AGENT_TOOLS,
    available_functions=AVAILABLE_FUNCTIONS,
)
if _added_local_schemas:
    print(f"Auto-registered {len(_added_local_schemas)} missing local tool schema(s).")


async def init_mcp_client():
    return await MCP_MANAGER.init_mcp_client(
        agent_tools=AGENT_TOOLS,
        available_functions=AVAILABLE_FUNCTIONS,
        register_or_update_tool_schema_fn=register_or_update_tool_schema,
    )


async def shutdown_mcp_client():
    await MCP_MANAGER.shutdown_mcp_client()


def get_mcp_runtime_status():
    return MCP_MANAGER.runtime_status()




