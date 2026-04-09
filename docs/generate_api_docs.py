"""Generate API_REFERENCE.md from the agent tool schemas."""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agent" / "agent"))

from tooling.schemas import AGENT_TOOLS
from tools import AVAILABLE_FUNCTIONS

CATEGORIES = {
    "Browser": ["browser_tabs_list", "browser_tab_select", "browser_close_blank_tabs"],
    "Filesystem": [
        "fs_list", "fs_read", "fs_read_batch", "fs_edit_lines", "fs_insert_lines",
        "fs_write", "fs_copy", "fs_move", "fs_delete", "fs_patch", "fs_search",
        "fs_analyze_file", "codebase_analyze",
    ],
    "Git": ["git_status", "git_diff", "git_log", "git_blame", "git_commit", "git_branch", "git_stash"],
    "Testing": ["generate_tests", "run_tests", "coverage_gaps"],
    "Refactoring": ["rename_symbol", "find_dead_code", "find_duplicates", "code_metrics"],
    "Snapshot & Rollback": ["snapshot_create", "snapshot_restore", "snapshot_list", "snapshot_diff"],
    "Vision": ["vision_encode", "vision_compare", "vision_describe_page"],
    "Documentation": ["generate_docstrings", "generate_changelog_entry", "doc_coverage"],
    "Memory": ["memory_log", "memory_search", "memory_get", "memory_promote", "memory_bootstrap", "memory_reindex"],
    "Web & OAuth": ["web_fetch", "oauth_set_profile", "oauth_get_token", "oauth_profiles"],
    "Workflow & Planning": ["reasoning_plan", "workflow_execute", "task_autopilot", "call_tool"],
    "Command Execution": ["run_command"],
    "Diagnostics": ["tool_catalog", "agent_health_report"],
    "Utility": ["calculate"],
}

schema_map = {}
for tool in AGENT_TOOLS:
    fn = tool.get("function", {})
    name = fn.get("name", "")
    if name:
        schema_map[name] = fn

lines = []
lines.append("# API Reference — Agent Tools\n")
lines.append(f"**{len(AVAILABLE_FUNCTIONS)} tools** across {len(CATEGORIES)} categories.\n")
lines.append("## Table of Contents\n")
for cat in CATEGORIES:
    anchor = cat.lower().replace(" & ", "-").replace(" ", "-")
    lines.append(f"- [{cat}](#{anchor})")
lines.append("")

for cat, names in CATEGORIES.items():
    lines.append(f"## {cat}\n")
    for name in names:
        fn = schema_map.get(name)
        if not fn:
            lines.append(f"### `{name}`\n")
            lines.append("Auto-registered tool. Use `tool_catalog` for full schema.\n")
            continue
        desc = fn.get("description", "")
        props = fn.get("parameters", {}).get("properties", {})
        required = set(fn.get("parameters", {}).get("required", []))
        lines.append(f"### `{name}`\n")
        lines.append(f"{desc}\n")
        if props:
            lines.append("| Parameter | Type | Required | Description |")
            lines.append("|---|---|---|---|")
            for pname, pinfo in props.items():
                ptype = pinfo.get("type", "any")
                pdesc = pinfo.get("description", "")
                preq = "Yes" if pname in required else "No"
                lines.append(f"| `{pname}` | {ptype} | {preq} | {pdesc} |")
            lines.append("")
        else:
            lines.append("*No parameters.*\n")

output = "\n".join(lines)
out_path = Path(__file__).resolve().parent / "API_REFERENCE.md"
out_path.write_text(output, encoding="utf-8")
print(f"Generated {out_path} with {len(AVAILABLE_FUNCTIONS)} tools documented.")
