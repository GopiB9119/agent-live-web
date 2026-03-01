import os
import json
import asyncio
import re
import inspect
from pathlib import Path
from dotenv import load_dotenv
from openai import AzureOpenAI
from tools import AGENT_TOOLS, AVAILABLE_FUNCTIONS, init_mcp_client, shutdown_mcp_client

# Load environment variables
load_dotenv()

# Initialize Azure OpenAI client
client = AzureOpenAI(
    api_key=os.environ.get("azure_key"),
    api_version=os.environ.get("azure_api_version", "2024-12-01-preview"),
    azure_endpoint=os.environ.get("azure_endpoint_uri")
)
# Fallback message if no API key is available during testing
if not client.api_key:
    print("WARNING: azure_key is not set in the .env file.")
    print("Please update your .env file with valid Azure OpenAI credentials.")

# For Azure, the 'model' parameter expects the deployment name
MODEL = os.environ.get("azure_deployment_name", "gpt-4o")
MAX_ITERATIONS = int(os.environ.get("AGENT_MAX_ITERATIONS", "10"))
MAX_HISTORY_MESSAGES = int(os.environ.get("AGENT_MAX_HISTORY_MESSAGES", "60"))
TOOL_TIMEOUT_SEC = float(os.environ.get("AGENT_TOOL_TIMEOUT_SEC", "180"))
MEMORY_AUTO_LOG = os.environ.get("AGENT_MEMORY_AUTO_LOG", "true").strip().lower() in {"1", "true", "yes", "on"}
MEMORY_PRIVATE_SESSION = os.environ.get("AGENT_PRIVATE_SESSION", "true").strip().lower() in {"1", "true", "yes", "on"}


def _load_system_prompt() -> str:
    prompt_file = Path(__file__).resolve().with_name("SYSTEM_PROMPT.md")
    default_prompt = (
        "You are an autonomous Live Web + Workspace Automation Agent. "
        "Use browser tools for interactive web tasks and fs tools for local codebase tasks. "
        "Never claim local folders are inaccessible when fs tools are available."
    )
    try:
        if prompt_file.exists():
            content = prompt_file.read_text(encoding="utf-8", errors="replace").strip()
            if content:
                return content
    except Exception:
        pass
    return default_prompt


def _looks_like_local_access_request(text: str) -> bool:
    lowered = text.lower()
    has_path_hint = bool(re.search(r"[a-zA-Z]:\\", text)) or "/" in text or "\\" in text
    has_scope_hint = any(word in lowered for word in ["codebase", "folder", "directory", "repo", "repository", "file", "path"])
    has_action_hint = any(word in lowered for word in ["see", "read", "open", "inspect", "check", "scan", "list", "show", "analyze"])
    return (has_path_hint and has_scope_hint) or (has_scope_hint and has_action_hint)


def _looks_like_local_access_refusal(text: str) -> bool:
    lowered = (text or "").lower()
    refusal_phrases = [
        "can't directly access",
        "cannot directly access",
        "can't access folders on your local",
        "cannot access folders on your local",
        "don't have access to your local",
        "do not have access to your local",
        "can't access your local computer",
        "can't directly interact with your local",
        "cannot directly interact with your local",
        "virtual workspace",
        "upload the files",
        "upload the necessary files",
    ]
    if any(phrase in lowered for phrase in refusal_phrases):
        return True
    return bool(
        re.search(r"\b(can('|no)t|cannot)\b.*\b(local|folder|directory|file system|computer)\b", lowered)
    )


def _extract_path_candidates(user_text: str):
    # Start from the first Windows drive-like sequence and progressively trim trailing words.
    match = re.search(r"[a-zA-Z]:\\", user_text)
    if not match:
        return []

    tail = user_text[match.start():].strip()
    tail = tail.strip("\"'`").strip()
    tail = tail.rstrip(".,;:!?")
    if not tail:
        return []

    candidates = []
    words = tail.split()
    for i in range(len(words), 0, -1):
        candidate = " ".join(words[:i]).strip()
        candidate = candidate.strip("\"'`").rstrip(".,;:!?")
        if candidate and candidate not in candidates:
            candidates.append(candidate)
    return candidates


async def _auto_local_access_fallback(user_prompt: str) -> str:
    fs_list_fn = AVAILABLE_FUNCTIONS.get("fs_list")
    fs_read_fn = AVAILABLE_FUNCTIONS.get("fs_read")
    if not fs_list_fn or not fs_read_fn:
        return "Local tools are not available in this session."

    path_candidates = _extract_path_candidates(user_prompt)
    if not path_candidates:
        path_candidates = ["."]

    for candidate in path_candidates:
        try:
            listing_raw = await fs_list_fn({"path": candidate, "recursive": False, "max_entries": 40, "include_hidden": False})
            listing = json.loads(listing_raw)
            if listing.get("status") == "ok":
                entries = listing.get("entries", [])
                head = entries[:12]
                rendered = "\n".join(f"- {item.get('path')} ({item.get('type')})" for item in head)
                return (
                    f"I can access your local workspace. I inspected `{candidate}`.\n"
                    f"Found {listing.get('count', 0)} entries.\n"
                    f"{rendered if rendered else '- (empty directory)'}\n"
                    "Tell me if you want recursive scan, specific file reads, or full codebase summary."
                )
        except Exception:
            pass

        try:
            read_raw = await fs_read_fn({"path": candidate, "max_chars": 4000})
            read_result = json.loads(read_raw)
            if read_result.get("status") == "ok":
                preview = read_result.get("content", "")
                return (
                    f"I can access your local workspace. I read `{candidate}`.\n"
                    f"Preview:\n{preview[:1200]}"
                )
        except Exception:
            pass

    attempted = ", ".join(f"`{p}`" for p in path_candidates[:5])
    return f"I attempted local access for {attempted}, but could not resolve a valid path. Share the exact folder path and I will inspect it."


async def _preflight_local_context(user_prompt: str):
    if not _looks_like_local_access_request(user_prompt):
        return None
    summary = await _auto_local_access_fallback(user_prompt)
    return summary


async def _memory_bootstrap_context():
    memory_bootstrap_fn = AVAILABLE_FUNCTIONS.get("memory_bootstrap")
    if not memory_bootstrap_fn:
        return None
    try:
        raw = await memory_bootstrap_fn(
            {
                "include_long_term": MEMORY_PRIVATE_SESSION,
                "max_chars": 24000,
            }
        )
        parsed = json.loads(raw)
        if parsed.get("status") == "ok" and parsed.get("content"):
            return parsed
    except Exception:
        return None
    return None


async def _memory_log_event(content: str, role: str, importance: int = 3, tags=None):
    if not MEMORY_AUTO_LOG:
        return
    memory_log_fn = AVAILABLE_FUNCTIONS.get("memory_log")
    if not memory_log_fn:
        return
    try:
        await memory_log_fn(
            {
                "content": content,
                "role": role,
                "importance": int(importance),
                "tags": tags or [],
            }
        )
    except Exception:
        # Do not block conversation on memory log failures.
        pass


def _trim_messages_for_context(messages):
    if len(messages) <= MAX_HISTORY_MESSAGES:
        return messages
    if not messages:
        return messages

    system = messages[0:1]
    tail = messages[-(MAX_HISTORY_MESSAGES - 1):]
    trimmed = system + tail
    if len(messages) > len(trimmed):
        trimmed.insert(
            1,
            {
                "role": "assistant",
                "content": f"[Context trimmed] Retained latest {len(trimmed)-1} messages for performance.",
            },
        )
    return trimmed

async def run_agent():
    """
    Runs the autonomous agent loop in an interactive chat session, supporting async MCP tools.
    """
    messages = [
        {"role": "system", "content": _load_system_prompt()}
    ]

    memory_context = await _memory_bootstrap_context()
    if memory_context:
        files = ", ".join(memory_context.get("files", []))
        messages.append(
            {
                "role": "system",
                "content": (
                    f"[Memory bootstrap loaded]\n"
                    f"Files: {files}\n"
                    f"{memory_context.get('content', '')}"
                ),
            }
        )
        print("  [Memory] Loaded startup memory context.")

    print("\n" + "="*50)
    print("Agent is ready! Type 'exit' or 'quit' to end the chat.")
    print("="*50 + "\n")

    while True:
        try:
            user_prompt = input("You: ")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting chat...")
            break
            
        if user_prompt.strip().lower() in ['exit', 'quit']:
            print("Goodbye!")
            break
            
        if not user_prompt.strip():
            continue

        # Add the user's new message to the history
        messages.append({"role": "user", "content": user_prompt})
        await _memory_log_event(user_prompt, role="user", importance=3, tags=["conversation"])

        # Preflight local workspace context when user references local paths/codebase.
        preflight_note = await _preflight_local_context(user_prompt)
        if preflight_note:
            messages.append({"role": "assistant", "content": f"[Local workspace preflight]\n{preflight_note}"})
            print("  [Preflight] Local workspace inspection completed.")

        iteration = 0
        while iteration < MAX_ITERATIONS:
            iteration += 1
            
            # Call the OpenAI API
            try:
                messages = _trim_messages_for_context(messages)
                response = client.chat.completions.create(
                    model=MODEL,
                    messages=messages,
                    tools=AGENT_TOOLS,
                    tool_choice="auto"  # The model decides whether to call a tool or not
                )
            except Exception as e:
                print(f"API Error: {e}")
                break

            response_message = response.choices[0].message
            
            # Check if the model decided to call a tool
            tool_calls = response_message.tool_calls

            if tool_calls:
                print(f"  [Agent is thinking... requested to use tool(s)]")
                # Append the assistant's request to the conversation history
                messages.append(response_message)
                
                # Execute each requested tool
                for tool_call in tool_calls:
                    function_name = tool_call.function.name
                    raw_args = tool_call.function.arguments
                    try:
                        function_args = json.loads(raw_args) if raw_args else {}
                        if not isinstance(function_args, dict):
                            function_args = {}
                    except Exception:
                        function_args = {}
                    
                    print(f"  -> Calling '{function_name}' with arguments: {function_args}")
                    
                    # Retrieve the actual Python function from our map
                    function_to_call = AVAILABLE_FUNCTIONS.get(function_name)
                    
                    if function_to_call:
                        # Execute the function (support both sync and async tools)
                        if inspect.iscoroutinefunction(function_to_call):
                            function_response = await asyncio.wait_for(
                                function_to_call(function_args),
                                timeout=TOOL_TIMEOUT_SEC
                            )
                        else:
                            function_response = function_to_call(**function_args)
                        
                        # Truncate print output if it's too long (like raw HTML)
                        print_resp = str(function_response)
                        print(f"  <- Result from tool: {print_resp[:300]}{'...' if len(print_resp) > 300 else ''}")
                    else:
                        function_response = f"Error: Tool '{function_name}' not found."
                        print(f"  <- {function_response}")
                    
                    # Append the tool's response to the conversation history
                    messages.append(
                        {
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": function_name,
                            "content": str(function_response),
                        }
                    )
                
                # The loop continues to the next iteration so the LLM can see the tool results
            else:
                # If there are no tool calls, the agent is finished and provides a standard text response
                final_answer = response_message.content
                if (
                    final_answer
                    and _looks_like_local_access_request(user_prompt)
                    and _looks_like_local_access_refusal(final_answer)
                ):
                    final_answer = await _auto_local_access_fallback(user_prompt)
                print(f"\nAgent: {final_answer}\n")
                await _memory_log_event(
                    final_answer,
                    role="assistant",
                    importance=3,
                    tags=["response"],
                )
                
                # Save the assistant's final response back into the history
                messages.append({"role": "assistant", "content": final_answer})
                break
                
        if iteration == MAX_ITERATIONS:
            print("\nAgent Error: Reached maximum internal steps without producing a final answer. Continuing to next turn...\n")

async def main():
    if client.api_key:
        print("Starting Autonomous Agent Interactive Chat...")
        try:
            # Initialize the MCP Client and connection before running
            await init_mcp_client()
            await run_agent()
        finally:
            await shutdown_mcp_client()
    else:
        print("Cannot run agent. Missing valid Azure OpenAI API key.")

if __name__ == "__main__":
    asyncio.run(main())

