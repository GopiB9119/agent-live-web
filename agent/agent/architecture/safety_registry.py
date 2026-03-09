from dataclasses import dataclass, field

try:
    from architecture.safety_types import ActionClass
except Exception:
    from .safety_types import ActionClass


@dataclass(frozen=True)
class ToolSafetyRule:
    action_class: str
    policy_tags: tuple[str, ...] = field(default_factory=tuple)


EXACT_RULES: dict[str, ToolSafetyRule] = {
    "calculate": ToolSafetyRule(ActionClass.READ_ONLY.value, ("utility",)),
    "browser_tabs_list": ToolSafetyRule(ActionClass.READ_ONLY.value, ("browser", "read")),
    "browser_tab_select": ToolSafetyRule(ActionClass.READ_ONLY.value, ("browser", "read")),
    "browser_close_blank_tabs": ToolSafetyRule(ActionClass.SCOPED_REVERSIBLE_WRITE.value, ("browser", "tab-cleanup")),
    "fs_list": ToolSafetyRule(ActionClass.READ_ONLY.value, ("workspace", "read")),
    "fs_read": ToolSafetyRule(ActionClass.READ_ONLY.value, ("workspace", "read")),
    "fs_read_batch": ToolSafetyRule(ActionClass.READ_ONLY.value, ("workspace", "read")),
    "fs_search": ToolSafetyRule(ActionClass.READ_ONLY.value, ("workspace", "read")),
    "fs_analyze_file": ToolSafetyRule(ActionClass.READ_ONLY.value, ("workspace", "read")),
    "codebase_analyze": ToolSafetyRule(ActionClass.READ_ONLY.value, ("workspace", "read")),
    "reasoning_plan": ToolSafetyRule(ActionClass.READ_ONLY.value, ("planning",)),
    "task_autopilot": ToolSafetyRule(ActionClass.READ_ONLY.value, ("planning", "analysis")),
    "tool_catalog": ToolSafetyRule(ActionClass.READ_ONLY.value, ("diagnostics", "read")),
    "agent_health_report": ToolSafetyRule(ActionClass.READ_ONLY.value, ("diagnostics", "read")),
    "fs_edit_lines": ToolSafetyRule(ActionClass.SCOPED_REVERSIBLE_WRITE.value, ("workspace", "write", "diffable")),
    "fs_insert_lines": ToolSafetyRule(ActionClass.SCOPED_REVERSIBLE_WRITE.value, ("workspace", "write", "diffable")),
    "fs_patch": ToolSafetyRule(ActionClass.SCOPED_REVERSIBLE_WRITE.value, ("workspace", "write", "diffable")),
    "fs_write": ToolSafetyRule(ActionClass.BROAD_LOCAL_WRITE.value, ("workspace", "write")),
    "fs_copy": ToolSafetyRule(ActionClass.BROAD_LOCAL_WRITE.value, ("workspace", "copy")),
    "fs_move": ToolSafetyRule(ActionClass.BROAD_LOCAL_WRITE.value, ("workspace", "move")),
    "fs_delete": ToolSafetyRule(ActionClass.DESTRUCTIVE.value, ("workspace", "delete")),
    "run_command": ToolSafetyRule(ActionClass.BROAD_LOCAL_WRITE.value, ("workspace", "command")),
    "oauth_set_profile": ToolSafetyRule(ActionClass.BROAD_LOCAL_WRITE.value, ("oauth", "secret-bearing")),
    "oauth_get_token": ToolSafetyRule(ActionClass.EXTERNAL_SIDE_EFFECT.value, ("oauth", "token")),
    "oauth_profiles": ToolSafetyRule(ActionClass.READ_ONLY.value, ("oauth", "read")),
    "web_fetch": ToolSafetyRule(ActionClass.READ_ONLY.value, ("network", "read")),
    "memory_get": ToolSafetyRule(ActionClass.READ_ONLY.value, ("memory", "read")),
    "memory_search": ToolSafetyRule(ActionClass.READ_ONLY.value, ("memory", "read")),
    "memory_bootstrap": ToolSafetyRule(ActionClass.READ_ONLY.value, ("memory", "read")),
    "memory_reindex": ToolSafetyRule(ActionClass.SCOPED_REVERSIBLE_WRITE.value, ("memory", "maintenance")),
    "memory_log": ToolSafetyRule(ActionClass.BROAD_LOCAL_WRITE.value, ("memory", "persistent")),
    "memory_promote": ToolSafetyRule(ActionClass.BROAD_LOCAL_WRITE.value, ("memory", "persistent")),
    "workflow_execute": ToolSafetyRule(ActionClass.BROAD_LOCAL_WRITE.value, ("workflow", "multi-step")),
    "call_tool": ToolSafetyRule(ActionClass.BROAD_LOCAL_WRITE.value, ("workflow", "indirect")),
}


READ_ONLY_BROWSER_TOOLS = {
    "browser_snapshot",
    "browser_tabs",
    "browser_console_messages",
    "browser_network_requests",
}
SAFE_BROWSER_ARTIFACT_TOOLS = {
    "browser_take_screenshot",
    "browser_pdf_save",
}
SCOPED_BROWSER_TOOLS = {
    "browser_navigate",
    "browser_navigate_back",
    "browser_wait_for",
    "browser_hover",
    "browser_close",
    "browser_install",
}
POTENTIAL_BROWSER_SIDE_EFFECT_TOOLS = {
    "browser_click",
    "browser_type",
    "browser_fill_form",
    "browser_select_option",
    "browser_press_key",
    "browser_drag",
    "browser_resize",
    "browser_file_upload",
}
DANGEROUS_BROWSER_TOOLS = {
    "browser_evaluate",
    "browser_run_code",
}


def get_tool_safety_rule(tool_name: str) -> ToolSafetyRule:
    clean_name = str(tool_name or "").strip()
    if clean_name in EXACT_RULES:
        return EXACT_RULES[clean_name]
    if clean_name in READ_ONLY_BROWSER_TOOLS:
        return ToolSafetyRule(ActionClass.READ_ONLY.value, ("browser", "read"))
    if clean_name in SAFE_BROWSER_ARTIFACT_TOOLS:
        return ToolSafetyRule(ActionClass.SCOPED_REVERSIBLE_WRITE.value, ("browser", "artifact"))
    if clean_name in SCOPED_BROWSER_TOOLS:
        return ToolSafetyRule(ActionClass.SCOPED_REVERSIBLE_WRITE.value, ("browser", "navigation"))
    if clean_name in POTENTIAL_BROWSER_SIDE_EFFECT_TOOLS:
        return ToolSafetyRule(ActionClass.SCOPED_REVERSIBLE_WRITE.value, ("browser", "interaction"))
    if clean_name in DANGEROUS_BROWSER_TOOLS:
        return ToolSafetyRule(ActionClass.DESTRUCTIVE.value, ("browser", "code-execution"))
    if clean_name.startswith("browser_"):
        return ToolSafetyRule(ActionClass.SCOPED_REVERSIBLE_WRITE.value, ("browser", "unknown"))
    return ToolSafetyRule(ActionClass.READ_ONLY.value, ("unknown",))
