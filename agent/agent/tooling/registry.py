import inspect


LOCAL_CALLABLE_NAMES = [
    "browser_tabs_list",
    "browser_tab_select",
    "browser_close_blank_tabs",
    "fs_list",
    "fs_read",
    "fs_read_batch",
    "fs_edit_lines",
    "fs_insert_lines",
    "fs_write",
    "fs_copy",
    "fs_move",
    "fs_delete",
    "fs_patch",
    "fs_search",
    "fs_analyze_file",
    "codebase_analyze",
    "reasoning_plan",
    "memory_log",
    "memory_search",
    "memory_get",
    "memory_promote",
    "memory_bootstrap",
    "memory_reindex",
    "tool_catalog",
    "agent_health_report",
    "workflow_execute",
    "task_autopilot",
    "oauth_set_profile",
    "oauth_get_token",
    "oauth_profiles",
    "run_command",
    "web_fetch",
    "call_tool",
]


def build_base_available_functions(calculate_fn):
    if not callable(calculate_fn):
        raise TypeError("calculate_fn must be callable")
    return {"calculate": calculate_fn}


def build_local_callable_registry(namespace: dict):
    if not isinstance(namespace, dict):
        raise TypeError("namespace must be a dict of symbol -> value")

    registry = {}
    missing = []
    for name in LOCAL_CALLABLE_NAMES:
        target = namespace.get(name)
        if callable(target):
            registry[name] = target
        else:
            missing.append(name)

    if missing:
        raise KeyError(f"Missing callable implementations for: {', '.join(missing)}")
    return registry


def tool_schema_name_set(agent_tools):
    names = set()
    for item in agent_tools:
        if not isinstance(item, dict):
            continue
        fn = item.get("function", {})
        name = fn.get("name")
        if isinstance(name, str) and name.strip():
            names.add(name.strip())
    return names


def register_or_update_tool_schema(agent_tools, name: str, description: str, parameters: dict):
    if not isinstance(name, str) or not name.strip():
        return

    clean_name = name.strip()
    schema_obj = {
        "type": "function",
        "function": {
            "name": clean_name,
            "description": description or f"Tool: {clean_name}",
            "parameters": parameters or {"type": "object", "properties": {}, "required": []},
        },
    }

    for idx, existing in enumerate(agent_tools):
        if not isinstance(existing, dict):
            continue
        fn = existing.get("function", {})
        if fn.get("name") == clean_name:
            agent_tools[idx] = schema_obj
            return

    agent_tools.append(schema_obj)


def auto_register_missing_local_tool_schemas(agent_tools, available_functions):
    existing = tool_schema_name_set(agent_tools)
    added = []

    for name, target in sorted(available_functions.items(), key=lambda item: item[0]):
        if not isinstance(name, str) or not name.strip():
            continue
        if name.startswith("_"):
            continue
        if name in existing:
            continue

        description = ""
        try:
            description = inspect.getdoc(target) or ""
        except Exception:
            description = ""
        if not description:
            description = f"Auto-registered local tool: {name}."

        register_or_update_tool_schema(
            agent_tools=agent_tools,
            name=name,
            description=description,
            parameters={"type": "object", "properties": {}, "required": []},
        )
        existing.add(name)
        added.append(name)

    return added
