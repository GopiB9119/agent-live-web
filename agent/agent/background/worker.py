"""
Background Worker — Executes a single task using the full agent tool chain.

Each worker:
1. Claims a task from the queue
2. Runs the AI conversation loop (same as interactive agent.py)
3. Streams progress back to the task file
4. Marks task completed or failed
5. Exits when done

Workers are spawned by the daemon, one per task.
"""
import asyncio
import inspect
import json
import sys
import time
from datetime import datetime
from pathlib import Path

# Ensure agent modules are importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from config import (
        client, MODEL, MODEL_PROVIDER, MODEL_SETUP_ERROR,
        MAX_ITERATIONS, TOOL_TIMEOUT_SEC, MAX_HISTORY_MESSAGES,
        RUNTIME_EXECUTION_GUIDE, MEMORY_PRIVATE_SESSION,
    )
except Exception:
    from ..config import (
        client, MODEL, MODEL_PROVIDER, MODEL_SETUP_ERROR,
        MAX_ITERATIONS, TOOL_TIMEOUT_SEC, MAX_HISTORY_MESSAGES,
        RUNTIME_EXECUTION_GUIDE, MEMORY_PRIVATE_SESSION,
    )

try:
    from tools import AGENT_TOOLS, AVAILABLE_FUNCTIONS, init_mcp_client, shutdown_mcp_client
except Exception:
    from ..tools import AGENT_TOOLS, AVAILABLE_FUNCTIONS, init_mcp_client, shutdown_mcp_client

try:
    from runtime_utils import (
        redact_sensitive_data as _redact_sensitive_data,
        redact_sensitive_text as _redact_sensitive_text,
        summarize_tool_outcome as _summarize_tool_outcome,
    )
except Exception:
    from ..runtime_utils import (
        redact_sensitive_data as _redact_sensitive_data,
        redact_sensitive_text as _redact_sensitive_text,
        summarize_tool_outcome as _summarize_tool_outcome,
    )

try:
    from background.task_queue import TaskQueue, TaskStatus
except Exception:
    from .task_queue import TaskQueue, TaskStatus


def _load_system_prompt() -> str:
    prompt_file = Path(__file__).resolve().parents[1] / "SYSTEM_PROMPT.md"
    default = "You are an autonomous background agent. Complete the task using available tools."
    try:
        if prompt_file.exists():
            content = prompt_file.read_text(encoding="utf-8", errors="replace").strip()
            if content:
                return content
    except Exception:
        pass
    return default


def _parse_tool_arguments(raw_args):
    if raw_args is None:
        return {}
    if isinstance(raw_args, dict):
        return raw_args
    import re
    text = str(raw_args).strip()
    if not text:
        return {}
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.IGNORECASE | re.DOTALL)
    candidates = [fenced.group(1).strip()] if fenced else []
    candidates.append(text)
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            continue
    return {}


async def execute_task(task: dict, queue: TaskQueue, use_mcp: bool = False) -> str:
    """
    Execute a task using the full agent tool chain.
    Returns the final result text.
    """
    task_id = task["task_id"]
    prompt = task["prompt"]
    max_iters = task.get("max_iterations", MAX_ITERATIONS)

    if client is None:
        queue.fail(task_id, f"Model client not configured ({MODEL_PROVIDER}): {MODEL_SETUP_ERROR}")
        return ""

    # Initialize MCP if requested
    if use_mcp:
        try:
            await init_mcp_client()
        except Exception as e:
            queue.update_progress(task_id, f"MCP init skipped: {e}")

    system_prompt = _load_system_prompt()
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "system", "content": RUNTIME_EXECUTION_GUIDE},
        {"role": "system", "content": f"[Background task mode] Task ID: {task_id}\nExecute autonomously. Report all progress."},
        {"role": "user", "content": prompt},
    ]

    tool_calls_log = []
    final_answer = ""

    try:
        for iteration in range(1, max_iters + 1):
            queue.update_progress(task_id, f"Iteration {iteration}/{max_iters}")

            # Trim messages if too long
            if len(messages) > MAX_HISTORY_MESSAGES:
                system_msgs = [m for m in messages[:6] if isinstance(m, dict) and m.get("role") == "system"]
                messages = system_msgs + messages[-(MAX_HISTORY_MESSAGES - len(system_msgs)):]

            try:
                # Run sync OpenAI call in a thread to not block the async event loop
                response = await asyncio.to_thread(
                    client.chat.completions.create,
                    model=MODEL,
                    messages=messages,
                    tools=AGENT_TOOLS,
                    tool_choice="auto",
                )
            except Exception as e:
                queue.fail(task_id, f"API error on iteration {iteration}: {e}")
                return ""

            response_message = response.choices[0].message
            tool_calls = response_message.tool_calls

            if tool_calls:
                messages.append(response_message)

                for tool_call in tool_calls:
                    function_name = tool_call.function.name
                    function_args = _parse_tool_arguments(tool_call.function.arguments)

                    queue.update_progress(
                        task_id,
                        f"Calling tool: {function_name}",
                        tool_name=function_name,
                    )

                    # Execute the tool
                    target = AVAILABLE_FUNCTIONS.get(function_name)
                    if not target:
                        tool_result = json.dumps({"status": "failed", "error": f"Tool not found: {function_name}"})
                    else:
                        try:
                            if inspect.iscoroutinefunction(target):
                                tool_result = await asyncio.wait_for(
                                    target(function_args),
                                    timeout=TOOL_TIMEOUT_SEC,
                                )
                            else:
                                try:
                                    tool_result = target(**function_args)
                                except TypeError:
                                    tool_result = target(function_args)
                        except asyncio.TimeoutError:
                            tool_result = json.dumps({"status": "failed", "error": f"Timeout after {TOOL_TIMEOUT_SEC}s"})
                        except Exception as e:
                            tool_result = json.dumps({"status": "failed", "error": str(e)})

                    sanitized = _redact_sensitive_text(str(tool_result), max_chars=8000)
                    tool_calls_log.append({
                        "iteration": iteration,
                        "tool": function_name,
                        "result_preview": sanitized[:300],
                    })

                    queue.update_progress(
                        task_id,
                        f"Tool result: {function_name}",
                        tool_name=function_name,
                        tool_result=sanitized[:500],
                    )

                    messages.append({
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": sanitized,
                    })

                continue  # Let model process tool results

            # No tool calls — model produced final answer
            final_answer = response_message.content or ""
            if not final_answer.strip():
                final_answer = "Task completed but no text summary was produced."
            break

        if not final_answer:
            final_answer = f"Reached max iterations ({max_iters}) without final answer."

        queue.complete(task_id, final_answer, tool_calls=tool_calls_log)
        return final_answer

    except Exception as e:
        queue.fail(task_id, f"Worker error: {e}")
        return ""
    finally:
        if use_mcp:
            try:
                await shutdown_mcp_client()
            except Exception:
                pass


async def run_worker(task_id: str, state_dir: str = None, use_mcp: bool = False):
    """Entry point for a worker process."""
    queue = TaskQueue(state_dir=Path(state_dir) if state_dir else None)
    task = queue.get(task_id)
    if not task:
        print(f"[Worker] Task not found: {task_id}")
        return
    if task["status"] != TaskStatus.RUNNING:
        print(f"[Worker] Task not in RUNNING state: {task['status']}")
        return

    print(f"[Worker] Starting task {task_id}: {task['prompt'][:100]}")
    result = await execute_task(task, queue, use_mcp=use_mcp)
    print(f"[Worker] Completed task {task_id}: {result[:200]}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run a background worker for a specific task")
    parser.add_argument("task_id", help="Task ID to execute")
    parser.add_argument("--state-dir", help="State directory path")
    parser.add_argument("--use-mcp", action="store_true", help="Initialize MCP client")
    args = parser.parse_args()
    asyncio.run(run_worker(args.task_id, args.state_dir, args.use_mcp))
