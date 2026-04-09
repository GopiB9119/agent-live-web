# API Reference — Agent Tools

**58 tools** across 14 categories.

## Table of Contents

- [Browser](#browser)
- [Filesystem](#filesystem)
- [Git](#git)
- [Testing](#testing)
- [Refactoring](#refactoring)
- [Snapshot & Rollback](#snapshot-rollback)
- [Vision](#vision)
- [Documentation](#documentation)
- [Memory](#memory)
- [Web & OAuth](#web-oauth)
- [Workflow & Planning](#workflow-planning)
- [Command Execution](#command-execution)
- [Diagnostics](#diagnostics)
- [Utility](#utility)

## Browser

### `browser_tabs_list`

Returns parsed browser tabs as JSON with index/title/url/current.

*No parameters.*

### `browser_tab_select`

Select a tab by index or by matching URL/title contains text.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `index` | number | No | Tab index to select. |
| `url_contains` | string | No | Select first tab whose URL contains this value. |
| `title_contains` | string | No | Select first tab whose title contains this value. |

### `browser_close_blank_tabs`

Closes extra about:blank tabs and keeps a real working tab selected.

*No parameters.*

## Filesystem

### `fs_list`

List files/directories in the workspace. Supports recursive listing.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `path` | string | No | Relative path inside workspace. |
| `recursive` | boolean | No | When true, include nested files. |
| `max_entries` | number | No | Maximum entries to return. |
| `include_hidden` | boolean | No | Include hidden files/folders. |

### `fs_read`

Read a text file from workspace.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `path` | string | Yes | Relative path inside workspace. |
| `encoding` | string | No | Text encoding. |
| `max_chars` | number | No | Max characters to return. |

### `fs_read_batch`

Read multiple text files in one call.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `paths` | array | Yes | List of file paths inside workspace. |
| `encoding` | string | No | Text encoding. |
| `max_chars_per_file` | number | No | Max chars per file. |
| `missing_ok` | boolean | No | Skip missing files without failing. |

### `fs_edit_lines`

Replace an inclusive line range in a text file.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `path` | string | Yes | Target file path. |
| `start_line` | number | Yes | 1-based start line. |
| `end_line` | number | Yes | 1-based end line (inclusive). |
| `replacement` | string | Yes | Replacement text block. |
| `encoding` | string | No | Text encoding. |
| `strict` | boolean | No | Fail if range is out of bounds. |
| `dry_run` | boolean | No | Preview changes without writing. |

### `fs_insert_lines`

Insert text at a specific 1-based line position.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `path` | string | Yes | Target file path. |
| `line` | number | Yes | 1-based line position (line_count+1 appends). |
| `content` | string | Yes | Text block to insert. |
| `encoding` | string | No | Text encoding. |
| `dry_run` | boolean | No | Preview changes without writing. |

### `fs_write`

Write or append text content to a workspace file.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `path` | string | Yes | Relative path inside workspace. |
| `content` | string | Yes | Text content to write. |
| `append` | boolean | No | Append instead of overwrite. |
| `encoding` | string | No | Text encoding. |

### `fs_copy`

Copy file or directory within workspace.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `source` | string | Yes | Source path inside workspace. |
| `destination` | string | Yes | Destination path inside workspace. |
| `overwrite` | boolean | No | Overwrite destination if it exists. |

### `fs_move`

Move or rename file/directory within workspace.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `source` | string | Yes | Source path inside workspace. |
| `destination` | string | Yes | Destination path inside workspace. |
| `overwrite` | boolean | No | Overwrite destination if it exists. |

### `fs_delete`

Delete file or directory from workspace.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `path` | string | Yes | Target path inside workspace. |
| `recursive` | boolean | No | Required for non-empty directories. |
| `missing_ok` | boolean | No | Do not fail when path is missing. |

### `fs_patch`

Apply structured find/replace edits to a file for safe refactors.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `path` | string | Yes | Target file path inside workspace. |
| `edits` | array | Yes | Ordered edit operations. |
| `encoding` | string | No | Text encoding. |
| `strict` | boolean | No | Fail if any edit has zero matches. |
| `create_if_missing` | boolean | No | Create file if missing. |
| `dry_run` | boolean | No | Preview changes without writing file. |

### `fs_search`

Search text pattern across files in workspace.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `pattern` | string | Yes | Text or regex pattern to find. |
| `path` | string | No | Root directory to search from. |
| `file_glob` | string | No | Glob filter like *.py or *.md. |
| `case_sensitive` | boolean | No | Case-sensitive search. |
| `max_results` | number | No | Maximum matches returned. |
| `regex` | boolean | No | Treat pattern as regex. |

### `fs_analyze_file`

Analyze a source/text file and return language, symbols, imports, and summary stats.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `path` | string | Yes | File path in workspace. |
| `encoding` | string | No | Text encoding. |
| `max_chars` | number | No | Maximum chars to parse. |
| `include_preview` | boolean | No | Include file head preview. |

### `codebase_analyze`

Analyze folder structure, language distribution, key files, and large files for understanding codebase.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `path` | string | No | Directory path in workspace. |
| `max_files` | number | No | Max files to scan. |
| `include_hidden` | boolean | No | Include hidden files and directories. |
| `top_n_large_files` | number | No | How many largest files to return. |

## Git

### `git_status`

Auto-registered local tool: git_status.

*No parameters.*

### `git_diff`

Auto-registered local tool: git_diff.

*No parameters.*

### `git_log`

Auto-registered local tool: git_log.

*No parameters.*

### `git_blame`

Auto-registered local tool: git_blame.

*No parameters.*

### `git_commit`

Auto-registered local tool: git_commit.

*No parameters.*

### `git_branch`

Auto-registered local tool: git_branch.

*No parameters.*

### `git_stash`

Auto-registered local tool: git_stash.

*No parameters.*

## Testing

### `generate_tests`

Auto-registered local tool: generate_tests.

*No parameters.*

### `run_tests`

Auto-registered local tool: run_tests.

*No parameters.*

### `coverage_gaps`

Auto-registered local tool: coverage_gaps.

*No parameters.*

## Refactoring

### `rename_symbol`

Auto-registered local tool: rename_symbol.

*No parameters.*

### `find_dead_code`

Auto-registered local tool: find_dead_code.

*No parameters.*

### `find_duplicates`

Auto-registered local tool: find_duplicates.

*No parameters.*

### `code_metrics`

Auto-registered local tool: code_metrics.

*No parameters.*

## Snapshot & Rollback

### `snapshot_create`

Auto-registered local tool: snapshot_create.

*No parameters.*

### `snapshot_restore`

Auto-registered local tool: snapshot_restore.

*No parameters.*

### `snapshot_list`

Auto-registered local tool: snapshot_list.

*No parameters.*

### `snapshot_diff`

Auto-registered local tool: snapshot_diff.

*No parameters.*

## Vision

### `vision_encode`

Auto-registered local tool: vision_encode.

*No parameters.*

### `vision_compare`

Auto-registered local tool: vision_compare.

*No parameters.*

### `vision_describe_page`

Auto-registered local tool: vision_describe_page.

*No parameters.*

## Documentation

### `generate_docstrings`

Auto-registered local tool: generate_docstrings.

*No parameters.*

### `generate_changelog_entry`

Auto-registered local tool: generate_changelog_entry.

*No parameters.*

### `doc_coverage`

Auto-registered local tool: doc_coverage.

*No parameters.*

## Memory

### `memory_log`

Append an important event/fact to today's daily memory log.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `content` | string | Yes | Memory text to store. |
| `role` | string | No | source role like user/assistant/system/event |
| `importance` | number | No | Importance 1-10 |
| `tags` | array | No | Optional tags |

### `memory_search`

Search daily and curated memory using hybrid lexical + embedding-style vector recall.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `query` | string | Yes | Search query. |
| `top_k` | number | No | How many results to return. |
| `include_long_term` | boolean | No | Include MEMORY.md. |
| `days_back` | number | No | How many daily logs to scan backward from today. |
| `use_semantic` | boolean | No | Enable vector similarity scoring. |
| `semantic_weight` | number | No | Weight for semantic score in final rank. |
| `lexical_weight` | number | No | Weight for lexical score in final rank. |
| `max_chunk_chars` | number | No | Chunk size used for lexical/vector memory matching. |
| `rebuild_index` | boolean | No | Force rebuilding memory vector index before search. |

### `memory_get`

Targeted read of a memory file range (daily file or MEMORY.md).

| Parameter | Type | Required | Description |
|---|---|---|---|
| `date` | string | No | Daily log date in YYYY-MM-DD format. |
| `file` | string | No | Explicit file path. Prefer memory files. |
| `start_line` | number | No | 1-based start line. |
| `end_line` | number | No | 1-based end line. |
| `max_chars` | number | No | Max chars to return. |

### `memory_promote`

Promote a critical fact to curated long-term MEMORY.md.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `fact` | string | Yes | Fact to persist long-term. |
| `importance` | number | No | Importance 1-10 |
| `tags` | array | No | Optional tags |

### `memory_bootstrap`

Load startup memory context from today/yesterday daily logs and optional long-term memory.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `include_long_term` | boolean | No | Include MEMORY.md |
| `max_chars` | number | No | Context size cap. |

### `memory_reindex`

Rebuild vector index for memory files to speed up semantic recall.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `include_long_term` | boolean | No | Include MEMORY.md. |
| `days_back` | number | No | How many daily logs to include. |
| `max_chunk_chars` | number | No | Chunk size for index entries. |
| `force_rebuild` | boolean | No | Ignore cached items and rebuild all vectors. |

## Web & OAuth

### `web_fetch`

Fetch live web URL content and optionally extract readable text.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `url` | string | Yes | HTTP/HTTPS URL to fetch. |
| `max_chars` | number | No | Max characters for body/text. |
| `extract_text` | boolean | No | Return tag-stripped text summary. |
| `timeout_sec` | number | No | Network timeout in seconds. |
| `allow_private_hosts` | boolean | No | Allow localhost/private/link-local hosts (off by default for SSRF protection). |
| `headers` | object | No | Optional request headers (string values). |
| `bearer_token` | string | No | Direct bearer token for Authorization header. Accepted for local use but never echoed back in tool output. |
| `oauth_profile` | string | No | OAuth profile name to auto-fetch bearer token. |
| `auth` | object | No | Optional auth object: {type:'bearer'|'oauth_profile', token|profile_name, force_refresh}. |

### `oauth_set_profile`

Create or update an in-memory OAuth profile for token retrieval.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `profile_name` | string | Yes | Unique profile name. |
| `token_url` | string | Yes | OAuth token endpoint URL. |
| `client_id` | string | Yes | OAuth client id. |
| `client_secret` | string | Yes | OAuth client secret. |
| `scope` | string | No | Optional scope string. |
| `audience` | string | No | Optional audience/resource value. |
| `grant_type` | string | No | Token grant type. Defaults to client_credentials. |
| `refresh_token` | string | No | Required only for refresh_token grant. |
| `extra_params` | object | No | Extra token form fields. |
| `timeout_sec` | number | No | HTTP timeout for token calls. |

### `oauth_get_token`

Get OAuth access token from profile or direct credentials (cached in memory).

| Parameter | Type | Required | Description |
|---|---|---|---|
| `profile_name` | string | No | Use a saved profile name. |
| `force_refresh` | boolean | No | Ignore cache and fetch new token. |
| `include_access_token` | boolean | No | Request raw token output. The runtime keeps raw token output disabled by default for secret hygiene; prefer oauth_profile-based usage. |
| `min_ttl_sec` | number | No | Minimum TTL for cache hit. |
| `token_url` | string | No | Direct mode token URL if no profile. |
| `client_id` | string | No | Direct mode client id. |
| `client_secret` | string | No | Direct mode client secret. |
| `scope` | string | No | Optional scope. |
| `audience` | string | No | Optional audience/resource. |
| `grant_type` | string | No | Grant type. |
| `refresh_token` | string | No | Refresh token for refresh_token grant. |
| `extra_params` | object | No | Extra token form fields. |
| `timeout_sec` | number | No | HTTP timeout for token calls. |

### `oauth_profiles`

List configured OAuth profiles and optionally remove one.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `action` | string | No | Action: list (default) or delete. |
| `profile_name` | string | No | Profile name for delete action. |

## Workflow & Planning

### `reasoning_plan`

Create a structured task plan (goal, assumptions, steps, risks) from user objective.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `goal` | string | Yes | Main objective to plan. |
| `context` | string | No | Optional context/constraints. |
| `max_steps` | number | No | Maximum plan steps. |

### `workflow_execute`

Execute a multi-step workflow of tool calls autonomously with per-step tracking.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `steps` | array | Yes | Ordered tool steps to execute. |
| `stop_on_error` | boolean | No | Stop workflow when a required step fails. |
| `include_artifact` | boolean | No | Include a sanitized reusable execution artifact and developer summary in the output. |
| `max_steps` | number | No | Safety cap for executed steps. |

### `task_autopilot`

Autonomous discovery runner: builds plan + codebase insights + file analyses for a goal.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `objective` | string | Yes | User objective to execute autonomously. |
| `path` | string | No | Workspace path for analysis. |
| `max_focus_files` | number | No | How many key files to inspect deeply. |
| `include_preview` | boolean | No | Include content previews in analysis. |
| `include_artifact` | boolean | No | Include a sanitized reusable task artifact and developer summary in the output. |

### `call_tool`

Invoke another registered tool by name with arguments object. Returned result text is sanitized for secret-bearing values.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `tool_name` | string | Yes | Exact tool function name. |
| `arguments` | object | No | Arguments object for that tool. |

## Command Execution

### `run_command`

Run a shell command in workspace and return exit code/stdout/stderr. Destructive patterns are blocked by default.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `command` | string | Yes | Shell command string. |
| `cwd` | string | No | Relative working directory inside workspace. |
| `timeout_sec` | number | No | Command timeout in seconds. |
| `security_mode` | string | No | Command policy mode: restricted (default) only allows safe read/check commands; permissive allows broader commands. |
| `confirm` | boolean | No | Explicit operator confirmation for dangerous command execution. Required with allow_dangerous=true. |
| `allow_dangerous` | boolean | No | Set true to bypass command safety blocklist. |

## Diagnostics

### `tool_catalog`

List all available tools with descriptions and whether they are currently callable.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `only_callable` | boolean | No | Return only callable tools. |

### `agent_health_report`

Run maintainability and registry integrity checks for agent/tool architecture.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `include_tools` | boolean | No | Include full tool-name lists in output. |
| `line_budgets` | object | No | Optional map of repo-relative file path -> max lines. |
| `fail_on_warn` | boolean | No | Return status=failed when warnings exist. |

## Utility

### `calculate`

Evaluates a mathematical expression and returns the result. Use this for all math-related queries.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `expression` | string | Yes | The math expression to evaluate, e.g., '453 * 89 + 12' |
