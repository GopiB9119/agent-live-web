# Source of truth for agent tool schemas.
AGENT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluates a mathematical expression and returns the result. Use this for all math-related queries.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The math expression to evaluate, e.g., '453 * 89 + 12'",
                    }
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "browser_tabs_list",
            "description": "Returns parsed browser tabs as JSON with index/title/url/current.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "browser_tab_select",
            "description": "Select a tab by index or by matching URL/title contains text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "index": {"type": "number", "description": "Tab index to select."},
                    "url_contains": {"type": "string", "description": "Select first tab whose URL contains this value."},
                    "title_contains": {"type": "string", "description": "Select first tab whose title contains this value."},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "browser_close_blank_tabs",
            "description": "Closes extra about:blank tabs and keeps a real working tab selected.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fs_list",
            "description": "List files/directories in the workspace. Supports recursive listing.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative path inside workspace.", "default": "."},
                    "recursive": {"type": "boolean", "description": "When true, include nested files.", "default": False},
                    "max_entries": {"type": "number", "description": "Maximum entries to return.", "default": 200},
                    "include_hidden": {"type": "boolean", "description": "Include hidden files/folders.", "default": False},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fs_read",
            "description": "Read a text file from workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative path inside workspace."},
                    "encoding": {"type": "string", "description": "Text encoding.", "default": "utf-8"},
                    "max_chars": {"type": "number", "description": "Max characters to return.", "default": 20000},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fs_write",
            "description": "Write or append text content to a workspace file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative path inside workspace."},
                    "content": {"type": "string", "description": "Text content to write."},
                    "append": {"type": "boolean", "description": "Append instead of overwrite.", "default": False},
                    "encoding": {"type": "string", "description": "Text encoding.", "default": "utf-8"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fs_read_batch",
            "description": "Read multiple text files in one call.",
            "parameters": {
                "type": "object",
                "properties": {
                    "paths": {
                        "type": "array",
                        "description": "List of file paths inside workspace.",
                        "items": {"type": "string"},
                    },
                    "encoding": {"type": "string", "description": "Text encoding.", "default": "utf-8"},
                    "max_chars_per_file": {"type": "number", "description": "Max chars per file.", "default": 12000},
                    "missing_ok": {"type": "boolean", "description": "Skip missing files without failing.", "default": True},
                },
                "required": ["paths"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fs_edit_lines",
            "description": "Replace an inclusive line range in a text file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Target file path."},
                    "start_line": {"type": "number", "description": "1-based start line."},
                    "end_line": {"type": "number", "description": "1-based end line (inclusive)."},
                    "replacement": {"type": "string", "description": "Replacement text block."},
                    "encoding": {"type": "string", "description": "Text encoding.", "default": "utf-8"},
                    "strict": {"type": "boolean", "description": "Fail if range is out of bounds.", "default": True},
                    "dry_run": {"type": "boolean", "description": "Preview changes without writing.", "default": False},
                },
                "required": ["path", "start_line", "end_line", "replacement"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fs_insert_lines",
            "description": "Insert text at a specific 1-based line position.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Target file path."},
                    "line": {"type": "number", "description": "1-based line position (line_count+1 appends)."},
                    "content": {"type": "string", "description": "Text block to insert."},
                    "encoding": {"type": "string", "description": "Text encoding.", "default": "utf-8"},
                    "dry_run": {"type": "boolean", "description": "Preview changes without writing.", "default": False},
                },
                "required": ["path", "line", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fs_copy",
            "description": "Copy file or directory within workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "source": {"type": "string", "description": "Source path inside workspace."},
                    "destination": {"type": "string", "description": "Destination path inside workspace."},
                    "overwrite": {"type": "boolean", "description": "Overwrite destination if it exists.", "default": False},
                },
                "required": ["source", "destination"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fs_move",
            "description": "Move or rename file/directory within workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "source": {"type": "string", "description": "Source path inside workspace."},
                    "destination": {"type": "string", "description": "Destination path inside workspace."},
                    "overwrite": {"type": "boolean", "description": "Overwrite destination if it exists.", "default": False},
                },
                "required": ["source", "destination"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fs_delete",
            "description": "Delete file or directory from workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Target path inside workspace."},
                    "recursive": {"type": "boolean", "description": "Required for non-empty directories.", "default": False},
                    "missing_ok": {"type": "boolean", "description": "Do not fail when path is missing.", "default": False},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fs_patch",
            "description": "Apply structured find/replace edits to a file for safe refactors.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Target file path inside workspace."},
                    "edits": {
                        "type": "array",
                        "description": "Ordered edit operations.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "find": {"type": "string", "description": "Text or regex pattern to find."},
                                "replace": {"type": "string", "description": "Replacement text."},
                                "regex": {"type": "boolean", "description": "Treat find as regex.", "default": False},
                                "count": {"type": "number", "description": "Max replacements for this edit. <=0 means all.", "default": 0},
                            },
                            "required": ["find", "replace"],
                        },
                    },
                    "encoding": {"type": "string", "description": "Text encoding.", "default": "utf-8"},
                    "strict": {"type": "boolean", "description": "Fail if any edit has zero matches.", "default": True},
                    "create_if_missing": {"type": "boolean", "description": "Create file if missing.", "default": False},
                    "dry_run": {"type": "boolean", "description": "Preview changes without writing file.", "default": False},
                },
                "required": ["path", "edits"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fs_search",
            "description": "Search text pattern across files in workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Text or regex pattern to find."},
                    "path": {"type": "string", "description": "Root directory to search from.", "default": "."},
                    "file_glob": {"type": "string", "description": "Glob filter like *.py or *.md.", "default": "*"},
                    "case_sensitive": {"type": "boolean", "description": "Case-sensitive search.", "default": False},
                    "max_results": {"type": "number", "description": "Maximum matches returned.", "default": 200},
                    "regex": {"type": "boolean", "description": "Treat pattern as regex.", "default": False},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fs_analyze_file",
            "description": "Analyze a source/text file and return language, symbols, imports, and summary stats.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path in workspace."},
                    "encoding": {"type": "string", "description": "Text encoding.", "default": "utf-8"},
                    "max_chars": {"type": "number", "description": "Maximum chars to parse.", "default": 200000},
                    "include_preview": {"type": "boolean", "description": "Include file head preview.", "default": True},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "codebase_analyze",
            "description": "Analyze folder structure, language distribution, key files, and large files for understanding codebase.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path in workspace.", "default": "."},
                    "max_files": {"type": "number", "description": "Max files to scan.", "default": 1200},
                    "include_hidden": {"type": "boolean", "description": "Include hidden files and directories.", "default": False},
                    "top_n_large_files": {"type": "number", "description": "How many largest files to return.", "default": 20},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "reasoning_plan",
            "description": "Create a structured task plan (goal, assumptions, steps, risks) from user objective.",
            "parameters": {
                "type": "object",
                "properties": {
                    "goal": {"type": "string", "description": "Main objective to plan."},
                    "context": {"type": "string", "description": "Optional context/constraints."},
                    "max_steps": {"type": "number", "description": "Maximum plan steps.", "default": 8},
                },
                "required": ["goal"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Run a shell command in workspace and return exit code/stdout/stderr. Destructive patterns are blocked by default.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command string."},
                    "cwd": {"type": "string", "description": "Relative working directory inside workspace.", "default": "."},
                    "timeout_sec": {"type": "number", "description": "Command timeout in seconds.", "default": 30},
                    "security_mode": {
                        "type": "string",
                        "description": "Command policy mode: restricted (default) only allows safe read/check commands; permissive allows broader commands.",
                        "default": "restricted",
                    },
                    "confirm": {
                        "type": "boolean",
                        "description": "Explicit operator confirmation for dangerous command execution. Required with allow_dangerous=true.",
                        "default": False,
                    },
                    "allow_dangerous": {"type": "boolean", "description": "Set true to bypass command safety blocklist.", "default": False},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "oauth_set_profile",
            "description": "Create or update an in-memory OAuth profile for token retrieval.",
            "parameters": {
                "type": "object",
                "properties": {
                    "profile_name": {"type": "string", "description": "Unique profile name."},
                    "token_url": {"type": "string", "description": "OAuth token endpoint URL."},
                    "client_id": {"type": "string", "description": "OAuth client id."},
                    "client_secret": {"type": "string", "description": "OAuth client secret."},
                    "scope": {"type": "string", "description": "Optional scope string."},
                    "audience": {"type": "string", "description": "Optional audience/resource value."},
                    "grant_type": {
                        "type": "string",
                        "description": "Token grant type. Defaults to client_credentials.",
                        "default": "client_credentials",
                    },
                    "refresh_token": {"type": "string", "description": "Required only for refresh_token grant."},
                    "extra_params": {"type": "object", "description": "Extra token form fields."},
                    "timeout_sec": {"type": "number", "description": "HTTP timeout for token calls.", "default": 20},
                },
                "required": ["profile_name", "token_url", "client_id", "client_secret"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "oauth_get_token",
            "description": "Get OAuth access token from profile or direct credentials (cached in memory).",
            "parameters": {
                "type": "object",
                "properties": {
                    "profile_name": {"type": "string", "description": "Use a saved profile name."},
                    "force_refresh": {"type": "boolean", "description": "Ignore cache and fetch new token.", "default": False},
                    "include_access_token": {
                        "type": "boolean",
                        "description": "Return full token in output. Keep false unless absolutely needed.",
                        "default": False,
                    },
                    "min_ttl_sec": {"type": "number", "description": "Minimum TTL for cache hit.", "default": 60},
                    "token_url": {"type": "string", "description": "Direct mode token URL if no profile."},
                    "client_id": {"type": "string", "description": "Direct mode client id."},
                    "client_secret": {"type": "string", "description": "Direct mode client secret."},
                    "scope": {"type": "string", "description": "Optional scope."},
                    "audience": {"type": "string", "description": "Optional audience/resource."},
                    "grant_type": {"type": "string", "description": "Grant type.", "default": "client_credentials"},
                    "refresh_token": {"type": "string", "description": "Refresh token for refresh_token grant."},
                    "extra_params": {"type": "object", "description": "Extra token form fields."},
                    "timeout_sec": {"type": "number", "description": "HTTP timeout for token calls.", "default": 20},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "oauth_profiles",
            "description": "List configured OAuth profiles and optionally remove one.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "description": "Action: list (default) or delete.",
                        "default": "list",
                    },
                    "profile_name": {"type": "string", "description": "Profile name for delete action."},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_fetch",
            "description": "Fetch live web URL content and optionally extract readable text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "HTTP/HTTPS URL to fetch."},
                    "max_chars": {"type": "number", "description": "Max characters for body/text.", "default": 50000},
                    "extract_text": {"type": "boolean", "description": "Return tag-stripped text summary.", "default": True},
                    "timeout_sec": {"type": "number", "description": "Network timeout in seconds.", "default": 20},
                    "allow_private_hosts": {
                        "type": "boolean",
                        "description": "Allow localhost/private/link-local hosts (off by default for SSRF protection).",
                        "default": False,
                    },
                    "headers": {"type": "object", "description": "Optional request headers (string values)."},
                    "bearer_token": {"type": "string", "description": "Direct bearer token for Authorization header."},
                    "oauth_profile": {"type": "string", "description": "OAuth profile name to auto-fetch bearer token."},
                    "auth": {
                        "type": "object",
                        "description": "Optional auth object: {type:'bearer'|'oauth_profile', token|profile_name, force_refresh}.",
                    },
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "call_tool",
            "description": "Invoke another registered tool by name with arguments object.",
            "parameters": {
                "type": "object",
                "properties": {
                    "tool_name": {"type": "string", "description": "Exact tool function name."},
                    "arguments": {"type": "object", "description": "Arguments object for that tool."},
                },
                "required": ["tool_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memory_log",
            "description": "Append an important event/fact to today's daily memory log.",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "Memory text to store."},
                    "role": {"type": "string", "description": "source role like user/assistant/system/event", "default": "event"},
                    "importance": {"type": "number", "description": "Importance 1-10", "default": 3},
                    "tags": {"type": "array", "items": {"type": "string"}, "description": "Optional tags"},
                },
                "required": ["content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memory_search",
            "description": "Search daily and curated memory using hybrid lexical + embedding-style vector recall.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query."},
                    "top_k": {"type": "number", "description": "How many results to return.", "default": 6},
                    "include_long_term": {"type": "boolean", "description": "Include MEMORY.md.", "default": True},
                    "days_back": {"type": "number", "description": "How many daily logs to scan backward from today.", "default": 14},
                    "use_semantic": {"type": "boolean", "description": "Enable vector similarity scoring.", "default": True},
                    "semantic_weight": {"type": "number", "description": "Weight for semantic score in final rank.", "default": 0.65},
                    "lexical_weight": {"type": "number", "description": "Weight for lexical score in final rank.", "default": 0.35},
                    "max_chunk_chars": {"type": "number", "description": "Chunk size used for lexical/vector memory matching.", "default": 1600},
                    "rebuild_index": {"type": "boolean", "description": "Force rebuilding memory vector index before search.", "default": False},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memory_get",
            "description": "Targeted read of a memory file range (daily file or MEMORY.md).",
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {"type": "string", "description": "Daily log date in YYYY-MM-DD format."},
                    "file": {"type": "string", "description": "Explicit file path. Prefer memory files."},
                    "start_line": {"type": "number", "description": "1-based start line.", "default": 1},
                    "end_line": {"type": "number", "description": "1-based end line.", "default": 200},
                    "max_chars": {"type": "number", "description": "Max chars to return.", "default": 20000},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memory_promote",
            "description": "Promote a critical fact to curated long-term MEMORY.md.",
            "parameters": {
                "type": "object",
                "properties": {
                    "fact": {"type": "string", "description": "Fact to persist long-term."},
                    "importance": {"type": "number", "description": "Importance 1-10", "default": 7},
                    "tags": {"type": "array", "items": {"type": "string"}, "description": "Optional tags"},
                },
                "required": ["fact"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memory_bootstrap",
            "description": "Load startup memory context from today/yesterday daily logs and optional long-term memory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "include_long_term": {"type": "boolean", "description": "Include MEMORY.md", "default": True},
                    "max_chars": {"type": "number", "description": "Context size cap.", "default": 24000},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memory_reindex",
            "description": "Rebuild vector index for memory files to speed up semantic recall.",
            "parameters": {
                "type": "object",
                "properties": {
                    "include_long_term": {"type": "boolean", "description": "Include MEMORY.md.", "default": True},
                    "days_back": {"type": "number", "description": "How many daily logs to include.", "default": 30},
                    "max_chunk_chars": {"type": "number", "description": "Chunk size for index entries.", "default": 1600},
                    "force_rebuild": {"type": "boolean", "description": "Ignore cached items and rebuild all vectors.", "default": True},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "tool_catalog",
            "description": "List all available tools with descriptions and whether they are currently callable.",
            "parameters": {
                "type": "object",
                "properties": {
                    "only_callable": {"type": "boolean", "description": "Return only callable tools.", "default": False}
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "agent_health_report",
            "description": "Run maintainability and registry integrity checks for agent/tool architecture.",
            "parameters": {
                "type": "object",
                "properties": {
                    "include_tools": {"type": "boolean", "description": "Include full tool-name lists in output.", "default": False},
                    "line_budgets": {
                        "type": "object",
                        "description": "Optional map of repo-relative file path -> max lines.",
                    },
                    "fail_on_warn": {"type": "boolean", "description": "Return status=failed when warnings exist.", "default": False},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "workflow_execute",
            "description": "Execute a multi-step workflow of tool calls autonomously with per-step tracking.",
            "parameters": {
                "type": "object",
                "properties": {
                    "steps": {
                        "type": "array",
                        "description": "Ordered tool steps to execute.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "tool_name": {"type": "string"},
                                "arguments": {"type": "object"},
                                "required": {"type": "boolean", "default": True},
                            },
                            "required": ["tool_name"],
                        },
                    },
                    "stop_on_error": {"type": "boolean", "description": "Stop workflow when a required step fails.", "default": True},
                    "max_steps": {"type": "number", "description": "Safety cap for executed steps.", "default": 30},
                },
                "required": ["steps"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "task_autopilot",
            "description": "Autonomous discovery runner: builds plan + codebase insights + file analyses for a goal.",
            "parameters": {
                "type": "object",
                "properties": {
                    "objective": {"type": "string", "description": "User objective to execute autonomously."},
                    "path": {"type": "string", "description": "Workspace path for analysis.", "default": "."},
                    "max_focus_files": {"type": "number", "description": "How many key files to inspect deeply.", "default": 6},
                    "include_preview": {"type": "boolean", "description": "Include content previews in analysis.", "default": False},
                },
                "required": ["objective"],
            },
        },
    },
]



