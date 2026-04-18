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
    from git_tools import GitManager
except Exception:
    from .git_tools import GitManager
try:
    from test_gen_tools import TestGenManager
except Exception:
    from .test_gen_tools import TestGenManager
try:
    from snapshot_tools import SnapshotManager
except Exception:
    from .snapshot_tools import SnapshotManager
try:
    from refactor_tools import RefactorManager
except Exception:
    from .refactor_tools import RefactorManager
try:
    from vision_tools import VisionManager
except Exception:
    from .vision_tools import VisionManager
try:
    from doc_tools import DocManager
except Exception:
    from .doc_tools import DocManager
try:
    from background_tools import BackgroundTaskManager
except Exception:
    from .background_tools import BackgroundTaskManager
try:
    from tooling.registry import (
        auto_register_missing_local_tool_schemas,
        build_base_available_functions,
        build_local_callable_registry,
        register_or_update_tool_schema,
    )
except Exception:
    from .tooling.registry import (
        auto_register_missing_local_tool_schemas,
        build_base_available_functions,
        build_local_callable_registry,
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
        redact_sensitive_data as _redact_sensitive_data,
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
        redact_sensitive_data as _redact_sensitive_data,
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
    "browser_hover",
    "browser_drag",
    "browser_wait_for",
}
STATE_CHANGE_TOOLS = {
    "browser_click",
    "browser_type",
    "browser_fill_form",
    "browser_select_option",
    "browser_press_key",
    "browser_hover",
    "browser_drag",
    "browser_file_upload",
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
)
DIAGNOSTICS_MANAGER = DiagnosticsManager(
    agent_tools_provider=lambda: AGENT_TOOLS,
    available_functions_provider=lambda: AVAILABLE_FUNCTIONS,
    resolve_workspace_path_fn=_resolve_workspace_path,
    to_bool_fn=_to_bool,
)
GIT_MANAGER = GitManager(
    workspace_root=WORKSPACE_ROOT,
    resolve_workspace_path_fn=_resolve_workspace_path,
)
TEST_GEN_MANAGER = TestGenManager(
    workspace_root=WORKSPACE_ROOT,
    resolve_workspace_path_fn=_resolve_workspace_path,
    fs_manager=FS_MANAGER,
)
SNAPSHOT_MANAGER = SnapshotManager(
    workspace_root=WORKSPACE_ROOT,
    resolve_workspace_path_fn=_resolve_workspace_path,
)
REFACTOR_MANAGER = RefactorManager(
    workspace_root=WORKSPACE_ROOT,
    resolve_workspace_path_fn=_resolve_workspace_path,
    fs_manager=FS_MANAGER,
)
VISION_MANAGER = VisionManager(
    workspace_root=WORKSPACE_ROOT,
    resolve_workspace_path_fn=_resolve_workspace_path,
)
DOC_MANAGER = DocManager(
    workspace_root=WORKSPACE_ROOT,
    resolve_workspace_path_fn=_resolve_workspace_path,
    fs_manager=FS_MANAGER,
)
BACKGROUND_MANAGER = BackgroundTaskManager()


# Define the local calculator backup tool
def calculate(expression: str) -> str:
    """Evaluates a basic math expression securely using AST parsing."""
    import ast
    import operator

    _OPS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    def _safe_eval(node):
        if isinstance(node, ast.Expression):
            return _safe_eval(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            left = _safe_eval(node.left)
            right = _safe_eval(node.right)
            if type(node.op) is ast.Pow and right > 100:
                raise ValueError("Exponent too large")
            return _OPS[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](_safe_eval(node.operand))
        raise ValueError(f"Unsupported expression node: {type(node).__name__}")

    try:
        tree = ast.parse(expression.strip(), mode="eval")
        result = _safe_eval(tree)
        return str(result)
    except (ValueError, SyntaxError, TypeError, ZeroDivisionError) as e:
        return f"Error evaluating expression: {e}"
    except Exception as e:
        return f"Error evaluating expression: {e}"


# Define the tools available to the OpenAI model (Starting with the local ones)
AVAILABLE_FUNCTIONS = build_base_available_functions(calculate)

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


# --- Git tools ---
async def git_status(kwargs_dict=None):
    return await GIT_MANAGER.git_status(kwargs_dict)

async def git_diff(kwargs_dict=None):
    return await GIT_MANAGER.git_diff(kwargs_dict)

async def git_log(kwargs_dict=None):
    return await GIT_MANAGER.git_log(kwargs_dict)

async def git_blame(kwargs_dict):
    return await GIT_MANAGER.git_blame(kwargs_dict)

async def git_commit(kwargs_dict):
    return await GIT_MANAGER.git_commit(kwargs_dict)

async def git_branch(kwargs_dict=None):
    return await GIT_MANAGER.git_branch(kwargs_dict)

async def git_stash(kwargs_dict=None):
    return await GIT_MANAGER.git_stash(kwargs_dict)


# --- Test generation tools ---
async def generate_tests(kwargs_dict):
    return await TEST_GEN_MANAGER.generate_tests(kwargs_dict)

async def run_tests(kwargs_dict=None):
    return await TEST_GEN_MANAGER.run_tests(kwargs_dict)

async def coverage_gaps(kwargs_dict):
    return await TEST_GEN_MANAGER.coverage_gaps(kwargs_dict)


# --- Snapshot/rollback tools ---
async def snapshot_create(kwargs_dict):
    return await SNAPSHOT_MANAGER.snapshot_create(kwargs_dict)

async def snapshot_restore(kwargs_dict):
    return await SNAPSHOT_MANAGER.snapshot_restore(kwargs_dict)

async def snapshot_list(kwargs_dict=None):
    return await SNAPSHOT_MANAGER.snapshot_list(kwargs_dict)

async def snapshot_diff(kwargs_dict):
    return await SNAPSHOT_MANAGER.snapshot_diff(kwargs_dict)


# --- Refactoring tools ---
async def rename_symbol(kwargs_dict):
    return await REFACTOR_MANAGER.rename_symbol(kwargs_dict)

async def find_dead_code(kwargs_dict=None):
    return await REFACTOR_MANAGER.find_dead_code(kwargs_dict)

async def find_duplicates(kwargs_dict=None):
    return await REFACTOR_MANAGER.find_duplicates(kwargs_dict)

async def code_metrics(kwargs_dict):
    return await REFACTOR_MANAGER.code_metrics(kwargs_dict)


# --- Vision tools ---
async def vision_encode(kwargs_dict):
    return await VISION_MANAGER.vision_encode(kwargs_dict)

async def vision_compare(kwargs_dict):
    return await VISION_MANAGER.vision_compare(kwargs_dict)

async def vision_describe_page(kwargs_dict):
    return await VISION_MANAGER.vision_describe_page(kwargs_dict)


# --- Documentation tools ---
async def generate_docstrings(kwargs_dict):
    return await DOC_MANAGER.generate_docstrings(kwargs_dict)

async def generate_changelog_entry(kwargs_dict=None):
    return await DOC_MANAGER.generate_changelog_entry(kwargs_dict)

async def doc_coverage(kwargs_dict=None):
    return await DOC_MANAGER.doc_coverage(kwargs_dict)


# --- Background task tools ---
async def bg_submit(kwargs_dict):
    return await BACKGROUND_MANAGER.bg_submit(kwargs_dict)

async def bg_status(kwargs_dict):
    return await BACKGROUND_MANAGER.bg_status(kwargs_dict)

async def bg_result(kwargs_dict):
    return await BACKGROUND_MANAGER.bg_result(kwargs_dict)

async def bg_cancel(kwargs_dict):
    return await BACKGROUND_MANAGER.bg_cancel(kwargs_dict)

async def bg_list(kwargs_dict=None):
    return await BACKGROUND_MANAGER.bg_list(kwargs_dict)


def _json_safe_response(payload, max_chars=50000):
    return json.dumps(_redact_sensitive_data(payload, max_chars=max_chars), ensure_ascii=True)


def _stringify_sanitized_tool_result(result):
    if isinstance(result, (dict, list)):
        return json.dumps(_redact_sensitive_data(result, max_chars=50000), ensure_ascii=True)
    return _redact_sensitive_text(str(result), max_chars=50000)


async def call_tool(kwargs_dict):
    kwargs = kwargs_dict or {}
    tool_name = str(kwargs.get("tool_name", "")).strip()
    arguments = kwargs.get("arguments", {}) or {}
    if not isinstance(arguments, dict):
        return _json_safe_response({"status": "failed", "error": "arguments must be an object"})
    if not tool_name:
        return _json_safe_response({"status": "failed", "error": "tool_name is required"})
    _blocked_from_call_tool = {"call_tool", "workflow_execute", "task_autopilot"}
    if tool_name in _blocked_from_call_tool:
        return _json_safe_response({"status": "failed", "error": f"Orchestration tool '{tool_name}' cannot be invoked through call_tool to prevent recursion"})

    target = AVAILABLE_FUNCTIONS.get(tool_name)
    if not target:
        return _json_safe_response({"status": "failed", "error": f"Tool not found: {tool_name}"})

    try:
        if inspect.iscoroutinefunction(target):
            result = await target(arguments)
        else:
            try:
                result = target(**arguments)
            except TypeError:
                result = target(arguments)
        return _json_safe_response({"status": "ok", "tool_name": tool_name, "result": _stringify_sanitized_tool_result(result)})
    except Exception as e:
        return _json_safe_response({"status": "failed", "tool_name": tool_name, "error": str(e)})


AVAILABLE_FUNCTIONS.update(build_local_callable_registry(globals()))

_added_local_schemas = auto_register_missing_local_tool_schemas(
    agent_tools=AGENT_TOOLS,
    available_functions=AVAILABLE_FUNCTIONS,
)
if _added_local_schemas:
    print(f"Auto-registered {len(_added_local_schemas)} missing local tool schema(s).")


async def init_mcp_client():
    await MCP_MANAGER.init_mcp_client(
        agent_tools=AGENT_TOOLS,
        available_functions=AVAILABLE_FUNCTIONS,
        register_or_update_tool_schema_fn=register_or_update_tool_schema,
    )


async def shutdown_mcp_client():
    await MCP_MANAGER.shutdown_mcp_client()




