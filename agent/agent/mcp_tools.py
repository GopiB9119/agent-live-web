import hashlib
import json
import os
import re
from contextlib import AsyncExitStack
from pathlib import Path
from urllib.parse import urlparse

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class MCPManager:
    """
    MCP session lifecycle + browser ownership/retry discipline manager.
    """

    TAB_LINE_RE = re.compile(
        r"^\s*-\s*(?P<index>\d+):\s*(?P<current>\(current\)\s*)?\[(?P<title>.*?)\]\((?P<url>.*?)\)\s*$"
    )

    def __init__(
        self,
        workspace_root: Path,
        retryable_tools=None,
        state_change_tools=None,
        ownership_skip_tools=None,
    ):
        self.workspace_root = Path(workspace_root).resolve()
        self.retryable_tools = set(retryable_tools or [])
        self.state_change_tools = set(state_change_tools or [])
        self.ownership_skip_tools = set(ownership_skip_tools or {"browser_tabs", "browser_close", "browser_install"})
        self.mcp_session = None
        self._mcp_exit_stack = None

    @staticmethod
    def _serialize_call_result(result) -> str:
        parts = []
        for item in getattr(result, "content", []) or []:
            text = getattr(item, "text", None)
            if text is not None:
                parts.append(str(text))
            elif hasattr(item, "model_dump"):
                parts.append(json.dumps(item.model_dump(), ensure_ascii=False))
            else:
                parts.append(str(item))
        if not parts and getattr(result, "structuredContent", None):
            parts.append(json.dumps(result.structuredContent, ensure_ascii=False))
        return "\n".join(parts).strip()

    @classmethod
    def _parse_tabs_text(cls, text: str):
        tabs = []
        for line in str(text).splitlines():
            match = cls.TAB_LINE_RE.match(line.strip())
            if not match:
                continue
            tabs.append(
                {
                    "index": int(match.group("index")),
                    "current": bool(match.group("current")),
                    "title": match.group("title") or "",
                    "url": match.group("url") or "",
                }
            )
        tabs.sort(key=lambda item: item["index"])
        return tabs

    @staticmethod
    def _host(url: str) -> str:
        try:
            return (urlparse(url).hostname or "").lower()
        except Exception:
            return ""

    @classmethod
    def _hosts_match(cls, expected_url: str, actual_url: str) -> bool:
        expected_host = cls._host(expected_url)
        actual_host = cls._host(actual_url)
        if not expected_host:
            return bool(actual_host)
        if expected_host == actual_host:
            return True
        return actual_host.endswith(f".{expected_host}")

    async def _call_mcp_tool_raw(self, tool_name: str, args=None):
        if self.mcp_session is None:
            return {
                "ok": False,
                "tool": tool_name,
                "args": args or {},
                "error": "MCP session is not connected.",
                "text": "",
            }

        payload = args or {}
        try:
            result = await self.mcp_session.call_tool(tool_name, arguments=payload)
            is_error = bool(getattr(result, "isError", False))
            text = self._serialize_call_result(result)
            return {
                "ok": not is_error,
                "tool": tool_name,
                "args": payload,
                "error": None if not is_error else text,
                "text": text,
                "result": result,
            }
        except Exception as e:
            return {
                "ok": False,
                "tool": tool_name,
                "args": payload,
                "error": str(e),
                "text": "",
            }

    async def _list_tabs_state(self):
        call = await self._call_mcp_tool_raw("browser_tabs", {"action": "list"})
        if not call["ok"]:
            return {"ok": False, "error": call["error"], "tabs": [], "current": None, "raw": call.get("text", "")}
        tabs = self._parse_tabs_text(call.get("text", ""))
        current = next((tab for tab in tabs if tab["current"]), None)
        return {"ok": True, "error": None, "tabs": tabs, "current": current, "raw": call.get("text", "")}

    async def _close_tab(self, index: int):
        return await self._call_mcp_tool_raw("browser_tabs", {"action": "close", "index": index})

    async def _select_tab(self, index: int):
        return await self._call_mcp_tool_raw("browser_tabs", {"action": "select", "index": index})

    async def _cleanup_blank_tabs(self, keep_current_blank=False):
        before = await self._list_tabs_state()
        if not before["ok"]:
            return before

        tabs = before["tabs"]
        non_blank = [tab for tab in tabs if tab["url"] and tab["url"] != "about:blank"]
        if not non_blank:
            return before

        current_index = before["current"]["index"] if before["current"] else None
        blanks = [tab for tab in tabs if tab["url"] == "about:blank"]
        for tab in sorted(blanks, key=lambda item: item["index"], reverse=True):
            if len(tabs) <= 1:
                break
            if keep_current_blank and tab["index"] == current_index:
                continue
            await self._close_tab(tab["index"])
            tabs = [entry for entry in tabs if entry["index"] != tab["index"]]

        return await self._list_tabs_state()

    async def _ensure_owned_working_tab(self, preserve_current_blank=False):
        tabs_state = await self._cleanup_blank_tabs(keep_current_blank=preserve_current_blank)
        if not tabs_state["ok"]:
            return tabs_state

        tabs = tabs_state["tabs"]
        if not tabs:
            await self._call_mcp_tool_raw("browser_tabs", {"action": "new"})
            tabs_state = await self._list_tabs_state()
            tabs = tabs_state["tabs"]
            if not tabs_state["ok"] or not tabs:
                return tabs_state

        if preserve_current_blank and tabs_state.get("current") and tabs_state["current"]["url"] == "about:blank":
            return tabs_state

        real_tabs = [tab for tab in tabs if tab["url"] and tab["url"] != "about:blank"]
        target = None
        if real_tabs:
            current_real = next((tab for tab in real_tabs if tab["current"]), None)
            target = current_real or real_tabs[-1]
        else:
            target = tabs_state.get("current") or tabs[-1]

        if target and not target["current"]:
            await self._select_tab(target["index"])
            tabs_state = await self._list_tabs_state()
        return tabs_state

    async def _capture_page_state(self, include_snapshot=False):
        tabs_state = await self._list_tabs_state()
        current = tabs_state["current"] if tabs_state["ok"] else None
        state = {
            "tabs_ok": tabs_state["ok"],
            "tabs_count": len(tabs_state["tabs"]) if tabs_state["ok"] else 0,
            "url": current["url"] if current else "",
            "title": current["title"] if current else "",
            "index": current["index"] if current else None,
            "snapshot_hash": None,
        }
        if include_snapshot:
            snap = await self._call_mcp_tool_raw("browser_snapshot", {})
            if snap["ok"]:
                state["snapshot_hash"] = hashlib.sha1(snap.get("text", "").encode("utf-8", errors="ignore")).hexdigest()
        return state

    async def _verify_step(self, tool_name: str, args: dict, before_state: dict, call_outcome: dict):
        if not call_outcome["ok"]:
            return {"ok": False, "reason": call_outcome.get("error", "Tool failed"), "details": {}}

        if tool_name == "browser_navigate":
            after = await self._capture_page_state(include_snapshot=False)
            actual_url = after.get("url", "")
            expected_url = str(args.get("url", "")).strip()
            ok = bool(actual_url and actual_url != "about:blank" and self._hosts_match(expected_url, actual_url))
            reason = (
                f"Expected host from '{expected_url}', current url is '{actual_url}'."
                if not ok
                else f"Navigation verified on '{actual_url}'."
            )
            return {"ok": ok, "reason": reason, "details": {"before": before_state, "after": after}}

        if tool_name == "browser_tabs":
            after = await self._list_tabs_state()
            action = args.get("action")
            if not after["ok"]:
                return {"ok": False, "reason": after.get("error", "Failed to list tabs"), "details": {"before": before_state}}
            if action == "select" and "index" in args:
                current_index = after["current"]["index"] if after["current"] else None
                ok = current_index == int(args["index"])
                reason = f"Tab select target={args['index']}, current={current_index}."
                return {"ok": ok, "reason": reason, "details": {"before": before_state, "after": after}}
            if action == "new":
                ok = after["tabs"] and len(after["tabs"]) >= before_state.get("tabs_count", 0)
                reason = f"Tab count is now {len(after['tabs'])}."
                return {"ok": ok, "reason": reason, "details": {"before": before_state, "after": after}}
            return {"ok": True, "reason": "Tab action completed.", "details": {"before": before_state, "after": after}}

        if tool_name in self.state_change_tools:
            after = await self._capture_page_state(include_snapshot=True)
            changed = False
            if before_state.get("url") != after.get("url"):
                changed = True
            if before_state.get("snapshot_hash") and after.get("snapshot_hash"):
                if before_state["snapshot_hash"] != after["snapshot_hash"]:
                    changed = True
            still_alive = bool(after.get("url") and after.get("url") != "about:blank")
            ok = changed or still_alive
            reason = "Page state changed after action." if changed else "Page remained stable but active tab is valid."
            return {"ok": ok, "reason": reason, "details": {"before": before_state, "after": after}}

        if tool_name == "browser_wait_for":
            return {"ok": True, "reason": "Wait condition satisfied by tool.", "details": {"before": before_state}}

        after = await self._capture_page_state(include_snapshot=False)
        ok = bool(after.get("url") or tool_name in {"browser_close", "browser_install"})
        reason = "Tool succeeded and browser context is reachable." if ok else "Browser context could not be verified."
        return {"ok": ok, "reason": reason, "details": {"before": before_state, "after": after}}

    @staticmethod
    def _format_step_response(tool_name: str, args: dict, attempt_count: int, verification: dict, outcome: dict, recovered: bool):
        payload = {
            "status": "ok" if (outcome.get("ok") and verification.get("ok")) else "failed",
            "tool": tool_name,
            "args": args,
            "attempts": attempt_count,
            "recovered": recovered,
            "verification": verification,
            "result": outcome.get("text", ""),
            "error": outcome.get("error"),
        }
        return json.dumps(payload, ensure_ascii=True)

    async def browser_tabs_list(self, _kwargs_dict=None):
        state = await self._list_tabs_state()
        if not state["ok"]:
            return json.dumps({"status": "failed", "error": state["error"]}, ensure_ascii=True)
        return json.dumps(
            {
                "status": "ok",
                "tabs": state["tabs"],
                "current": state["current"],
            },
            ensure_ascii=True,
        )

    async def browser_tab_select(self, kwargs_dict):
        if self.mcp_session is None:
            return json.dumps({"status": "failed", "error": "MCP session is not connected."}, ensure_ascii=True)

        kwargs = kwargs_dict or {}
        target_index = kwargs.get("index")
        url_contains = str(kwargs.get("url_contains", "")).strip().lower()
        title_contains = str(kwargs.get("title_contains", "")).strip().lower()

        tabs_state = await self._list_tabs_state()
        if not tabs_state["ok"]:
            return json.dumps({"status": "failed", "error": tabs_state["error"]}, ensure_ascii=True)

        tabs = tabs_state["tabs"]
        target = None
        if target_index is not None:
            try:
                requested = int(target_index)
            except Exception:
                return json.dumps({"status": "failed", "error": f"Invalid index: {target_index}"}, ensure_ascii=True)
            target = next((tab for tab in tabs if tab["index"] == requested), None)
        else:
            for tab in tabs:
                url_match = url_contains and url_contains in tab["url"].lower()
                title_match = title_contains and title_contains in tab["title"].lower()
                if url_match or title_match:
                    target = tab
                    break

        if target is None:
            return json.dumps(
                {"status": "failed", "error": "No matching tab found.", "tabs": tabs},
                ensure_ascii=True,
            )

        selected = await self._select_tab(target["index"])
        if not selected["ok"]:
            return json.dumps({"status": "failed", "error": selected["error"]}, ensure_ascii=True)

        final_state = await self._list_tabs_state()
        return json.dumps(
            {"status": "ok", "selected_index": target["index"], "current": final_state.get("current"), "tabs": final_state.get("tabs", [])},
            ensure_ascii=True,
        )

    async def browser_close_blank_tabs(self, _kwargs_dict=None):
        state = await self._cleanup_blank_tabs()
        if not state["ok"]:
            return json.dumps({"status": "failed", "error": state["error"]}, ensure_ascii=True)
        return json.dumps({"status": "ok", "tabs": state["tabs"], "current": state["current"]}, ensure_ascii=True)

    async def init_mcp_client(self, agent_tools, available_functions, register_or_update_tool_schema_fn):
        if self.mcp_session is not None:
            return

        if not callable(register_or_update_tool_schema_fn):
            raise TypeError("register_or_update_tool_schema_fn must be callable")

        project_root = self.workspace_root
        default_runtime_root = (
            Path(os.environ["LOCALAPPDATA"]) / "PlaywrightMCP"
            if os.environ.get("LOCALAPPDATA")
            else project_root / ".playwright-mcp"
        )
        owner_file = Path(os.environ.get("PLAYWRIGHT_MCP_OWNER_FILE", str(project_root / ".playwright-mcp" / "active-owner.txt")))
        user_data_dir = Path(os.environ.get("PLAYWRIGHT_MCP_USER_DATA_DIR", str(default_runtime_root / "edge-profile")))
        output_dir = Path(os.environ.get("PLAYWRIGHT_MCP_OUTPUT_DIR", str(default_runtime_root / "output")))
        mcp_env = os.environ.copy()
        mcp_env.update(
            {
                "PLAYWRIGHT_MCP_OWNER": "python",
                "PLAYWRIGHT_MCP_OWNER_FILE": str(owner_file),
                "PLAYWRIGHT_MCP_PERSIST_PROFILE": "true",
                "PLAYWRIGHT_MCP_SAVE_SESSION": "false",
                "PLAYWRIGHT_MCP_SAVE_TRACE": "false",
                "PLAYWRIGHT_MCP_OUTPUT_MODE": "stdout",
                "PLAYWRIGHT_MCP_SNAPSHOT_MODE": "incremental",
                "PLAYWRIGHT_MCP_CONSOLE_LEVEL": "error",
                "PLAYWRIGHT_MCP_TIMEOUT_ACTION_MS": "12000",
                "PLAYWRIGHT_MCP_TIMEOUT_NAVIGATION_MS": "60000",
                "PLAYWRIGHT_MCP_SHARED_BROWSER_CONTEXT": "true",
                "PLAYWRIGHT_MCP_BLOCK_SERVICE_WORKERS": "true",
                "PLAYWRIGHT_MCP_BLOCKED_ORIGINS": "http://127.0.0.1;http://localhost;http://[::1];https://127.0.0.1;https://localhost;https://[::1];http://169.254.169.254;http://169.254.170.2",
                "PLAYWRIGHT_MCP_USER_DATA_DIR": str(user_data_dir),
                "PLAYWRIGHT_MCP_OUTPUT_DIR": str(output_dir),
            }
        )

        mcp_launcher = project_root / "playwright-edge-mcp.js"
        if not mcp_launcher.exists():
            raise FileNotFoundError(f"MCP launcher not found: {mcp_launcher}")

        server_params = StdioServerParameters(command="node", args=[str(mcp_launcher)], env=mcp_env)
        print("Connecting to MCP Playwright server...")

        try:
            self._mcp_exit_stack = AsyncExitStack()
            read, write = await self._mcp_exit_stack.enter_async_context(stdio_client(server_params))
            self.mcp_session = await self._mcp_exit_stack.enter_async_context(ClientSession(read, write))
            await self.mcp_session.initialize()

            mcp_tools = await self.mcp_session.list_tools()
            print(f"Connected to MCP server! Found {len(mcp_tools.tools)} tools.")

            for tool in mcp_tools.tools:
                properties = {}
                required = []
                if getattr(tool, "inputSchema", None) and "properties" in tool.inputSchema:
                    properties = tool.inputSchema["properties"]
                    required = tool.inputSchema.get("required", [])

                register_or_update_tool_schema_fn(
                    agent_tools=agent_tools,
                    name=tool.name,
                    description=tool.description or f"Tool provided by MCP: {tool.name}",
                    parameters={"type": "object", "properties": properties, "required": required},
                )

                async def execute_mcp_tool(kwargs_dict, current_tool=tool.name):
                    args = kwargs_dict or {}
                    print(f"Executing MCP Tool --> {current_tool} with arguments: {args}")

                    if current_tool.startswith("browser_") and current_tool not in self.ownership_skip_tools:
                        ownership = await self._ensure_owned_working_tab(preserve_current_blank=(current_tool == "browser_navigate"))
                        if not ownership.get("ok", False):
                            return json.dumps(
                                {
                                    "status": "failed",
                                    "tool": current_tool,
                                    "error": f"Tab ownership check failed: {ownership.get('error', 'unknown error')}",
                                },
                                ensure_ascii=True,
                            )

                    before = await self._capture_page_state(include_snapshot=current_tool in self.state_change_tools)
                    first = await self._call_mcp_tool_raw(current_tool, args)
                    first_verification = await self._verify_step(current_tool, args, before, first)
                    if first["ok"] and first_verification["ok"]:
                        return self._format_step_response(current_tool, args, 1, first_verification, first, recovered=False)

                    if current_tool not in self.retryable_tools:
                        return self._format_step_response(current_tool, args, 1, first_verification, first, recovered=False)

                    await self._call_mcp_tool_raw("browser_wait_for", {"time": 2})
                    second_before = await self._capture_page_state(include_snapshot=current_tool in self.state_change_tools)
                    second = await self._call_mcp_tool_raw(current_tool, args)
                    second_verification = await self._verify_step(current_tool, args, second_before, second)
                    return self._format_step_response(
                        current_tool,
                        args,
                        2,
                        second_verification,
                        second,
                        recovered=second["ok"] and second_verification["ok"],
                    )

                available_functions[tool.name] = execute_mcp_tool

            await self._ensure_owned_working_tab()
            print("MCP tools successfully loaded into AGENT_TOOLS.")

        except Exception as e:
            print(f"Failed to connect to MCP server: {e}")
            print("Agent will proceed without external MCP tools.")
            if self._mcp_exit_stack is not None:
                try:
                    await self._mcp_exit_stack.aclose()
                except Exception:
                    pass
            self._mcp_exit_stack = None
            self.mcp_session = None

    async def shutdown_mcp_client(self):
        if self._mcp_exit_stack is not None:
            try:
                await self._mcp_exit_stack.aclose()
            except Exception:
                pass
        self._mcp_exit_stack = None
        self.mcp_session = None
