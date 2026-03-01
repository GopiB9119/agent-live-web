import os
import json
import asyncio
import inspect
import hashlib
import re
import math
import fnmatch
import subprocess
import shutil
import time
import html as html_lib
import urllib.request
import urllib.error
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import urlparse
from contextlib import AsyncExitStack
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


TAB_LINE_RE = re.compile(
    r"^\s*-\s*(?P<index>\d+):\s*(?P<current>\(current\)\s*)?\[(?P<title>.*?)\]\((?P<url>.*?)\)\s*$"
)
RETRYABLE_TOOLS = {
    "browser_navigate",
    "browser_click",
    "browser_type",
    "browser_fill_form",
    "browser_select_option",
    "browser_press_key",
    "browser_wait_for",
}
STATE_CHANGE_TOOLS = {
    "browser_click",
    "browser_type",
    "browser_fill_form",
    "browser_select_option",
    "browser_press_key",
}
OWNERSHIP_SKIP_TOOLS = {"browser_tabs", "browser_close", "browser_install"}
WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
NOISE_DIR_NAMES = {".git", "node_modules", "__pycache__", ".venv", "venv", ".mypy_cache", ".pytest_cache"}
BINARY_SUFFIXES = {
    ".pyc", ".pyd", ".so", ".dll", ".exe", ".bin", ".dat", ".db",
    ".zip", ".gz", ".7z", ".rar", ".tar",
    ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".ico",
    ".mp3", ".wav", ".ogg", ".mp4", ".mkv", ".avi", ".mov",
    ".pdf", ".woff", ".woff2", ".ttf", ".otf",
}
MEMORY_DIR = WORKSPACE_ROOT / "memory"
LONG_TERM_MEMORY_FILE = WORKSPACE_ROOT / "MEMORY.md"
MEMORY_VECTOR_INDEX_FILE = MEMORY_DIR / ".vector_index.json"
MEMORY_VECTOR_DIM = 192


def _is_path_within_root(path_obj: Path, root_obj: Path) -> bool:
    try:
        path_obj.relative_to(root_obj)
        return True
    except ValueError:
        return False


def _resolve_workspace_path(raw_path: str, must_exist: bool = False) -> Path:
    if not raw_path:
        raise ValueError("Path is required.")

    candidate = Path(raw_path)
    if not candidate.is_absolute():
        candidate = (WORKSPACE_ROOT / candidate).resolve()
    else:
        candidate = candidate.resolve()

    if not _is_path_within_root(candidate, WORKSPACE_ROOT.resolve()):
        raise ValueError(f"Path is outside workspace root: {candidate}")

    if must_exist and not candidate.exists():
        raise FileNotFoundError(f"Path does not exist: {candidate}")

    return candidate


def _strip_html_to_text(raw_html: str) -> str:
    no_script = re.sub(r"<script\b[^<]*(?:(?!</script>)<[^<]*)*</script>", " ", raw_html, flags=re.IGNORECASE | re.DOTALL)
    no_style = re.sub(r"<style\b[^<]*(?:(?!</style>)<[^<]*)*</style>", " ", no_script, flags=re.IGNORECASE | re.DOTALL)
    no_tags = re.sub(r"<[^>]+>", " ", no_style)
    unescaped = html_lib.unescape(no_tags)
    normalized = re.sub(r"\s+", " ", unescaped).strip()
    return normalized


def _extract_title(raw_html: str) -> str:
    match = re.search(r"<title[^>]*>(.*?)</title>", raw_html, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return ""
    return re.sub(r"\s+", " ", html_lib.unescape(match.group(1))).strip()


def _language_from_path(path_obj: Path) -> str:
    suffix = path_obj.suffix.lower()
    mapping = {
        ".py": "python",
        ".js": "javascript",
        ".cjs": "javascript",
        ".mjs": "javascript",
        ".ts": "typescript",
        ".tsx": "typescript",
        ".jsx": "javascript",
        ".json": "json",
        ".md": "markdown",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".html": "html",
        ".css": "css",
        ".scss": "scss",
        ".txt": "text",
        ".toml": "toml",
        ".xml": "xml",
        ".sh": "shell",
        ".ps1": "powershell",
        ".java": "java",
        ".go": "go",
        ".rs": "rust",
        ".rb": "ruby",
        ".php": "php",
    }
    return mapping.get(suffix, "unknown")


def _extract_python_symbols(content: str):
    imports = []
    functions = []
    classes = []

    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("import "):
            imports.append(stripped)
        elif stripped.startswith("from ") and " import " in stripped:
            imports.append(stripped)

    functions = re.findall(r"^\s*def\s+([A-Za-z_]\w*)\s*\(", content, flags=re.MULTILINE)
    classes = re.findall(r"^\s*class\s+([A-Za-z_]\w*)\b", content, flags=re.MULTILINE)
    return {"imports": imports[:120], "functions": functions[:200], "classes": classes[:200]}


def _extract_js_ts_symbols(content: str):
    imports = re.findall(r"^\s*import\s+.+$", content, flags=re.MULTILINE)
    classes = re.findall(r"^\s*class\s+([A-Za-z_]\w*)\b", content, flags=re.MULTILINE)
    fn_decl = re.findall(r"^\s*(?:export\s+)?function\s+([A-Za-z_]\w*)\s*\(", content, flags=re.MULTILINE)
    fn_expr = re.findall(
        r"^\s*(?:export\s+)?(?:const|let|var)\s+([A-Za-z_]\w*)\s*=\s*(?:async\s*)?\([^)]*\)\s*=>",
        content,
        flags=re.MULTILINE,
    )
    return {
        "imports": imports[:120],
        "functions": (fn_decl + fn_expr)[:240],
        "classes": classes[:200],
    }


def _analyze_content(path_obj: Path, content: str):
    language = _language_from_path(path_obj)
    lines = content.splitlines()
    line_count = len(lines)

    if language == "python":
        symbol_data = _extract_python_symbols(content)
    elif language in {"javascript", "typescript"}:
        symbol_data = _extract_js_ts_symbols(content)
    else:
        symbol_data = {"imports": [], "functions": [], "classes": []}

    return {
        "language": language,
        "line_count": line_count,
        "imports_count": len(symbol_data["imports"]),
        "functions_count": len(symbol_data["functions"]),
        "classes_count": len(symbol_data["classes"]),
        "imports": symbol_data["imports"][:40],
        "functions": symbol_data["functions"][:80],
        "classes": symbol_data["classes"][:80],
    }


def _coerce_tool_result_to_dict(raw_result):
    if isinstance(raw_result, dict):
        return raw_result
    text = str(raw_result)
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    return {"status": "ok", "raw": text}


def _is_noise_path(path_obj: Path) -> bool:
    return any(part in NOISE_DIR_NAMES for part in path_obj.parts)


def _is_probably_text_source(path_obj: Path) -> bool:
    if _is_noise_path(path_obj):
        return False
    if path_obj.suffix.lower() in BINARY_SUFFIXES:
        return False
    return True


def _ensure_memory_paths():
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    if not LONG_TERM_MEMORY_FILE.exists():
        LONG_TERM_MEMORY_FILE.write_text("# Curated Long-Term Memory\n\n", encoding="utf-8")


def _daily_memory_file(target_date: datetime) -> Path:
    _ensure_memory_paths()
    return MEMORY_DIR / f"{target_date.strftime('%Y-%m-%d')}.md"


def _append_daily_memory(content: str, role: str = "event", importance: int = 3, tags=None):
    tags = tags or []
    _ensure_memory_paths()
    now = datetime.now()
    file_path = _daily_memory_file(now)
    if not file_path.exists():
        file_path.write_text(f"# Daily Memory Log - {now.strftime('%Y-%m-%d')}\n\n", encoding="utf-8")
    safe_importance = max(1, min(int(importance), 10))
    tag_text = ",".join(sorted({str(tag).strip() for tag in tags if str(tag).strip()}))
    header = f"## {now.isoformat(timespec='seconds')} | role:{role} | importance:{safe_importance}"
    if tag_text:
        header += f" | tags:{tag_text}"
    body = str(content).strip()
    entry = f"{header}\n- {body}\n\n"
    with file_path.open("a", encoding="utf-8", errors="replace", newline="") as fh:
        fh.write(entry)
    return file_path


def _read_file_lines(path_obj: Path):
    text = path_obj.read_text(encoding="utf-8", errors="replace")
    return text, text.splitlines()


def _tokenize_for_memory(text: str):
    return re.findall(r"[a-zA-Z0-9_]+", text.lower())


def _memory_chunk_score(query_tokens, chunk_text: str, recency_boost: float = 0.0):
    chunk_tokens = _tokenize_for_memory(chunk_text)
    if not chunk_tokens:
        return 0.0
    chunk_set = set(chunk_tokens)
    overlap = sum(1 for token in query_tokens if token in chunk_set)
    density = overlap / max(1, len(chunk_set))
    importance_match = re.search(r"importance:(\d+)", chunk_text)
    importance = int(importance_match.group(1)) if importance_match else 3
    importance_boost = min(importance, 10) * 0.15
    return overlap + density + importance_boost + recency_boost


def _memory_file_recency_boost(path_obj: Path):
    if path_obj == LONG_TERM_MEMORY_FILE:
        return 1.0
    try:
        date_str = path_obj.stem
        file_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        today = datetime.now().date()
        if file_date == today:
            return 2.0
        if file_date == (today - timedelta(days=1)):
            return 1.5
        days_old = (today - file_date).days
        return max(0.1, 1.0 / (1 + days_old))
    except Exception:
        return 0.5


def _memory_chunks_from_text(text: str, max_chunk_chars: int = 1600):
    # Split by headings first, then enforce size limits.
    sections = re.split(r"(?m)^##\s+", text)
    chunks = []
    for section in sections:
        section = section.strip()
        if not section:
            continue
        if len(section) <= max_chunk_chars:
            chunks.append(section)
            continue
        start = 0
        while start < len(section):
            chunks.append(section[start:start + max_chunk_chars])
            start += max_chunk_chars
    return chunks


def _stable_token_hash(token: str) -> int:
    digest = hashlib.sha1(token.encode("utf-8", errors="ignore")).hexdigest()
    return int(digest[:8], 16)


def _normalize_vector(values):
    if not values:
        return []
    magnitude = math.sqrt(sum(float(v) * float(v) for v in values))
    if magnitude <= 1e-12:
        return [0.0 for _ in values]
    return [float(v) / magnitude for v in values]


def _hash_embed_text(text: str, dim: int = MEMORY_VECTOR_DIM):
    # Hashing-based embedding fallback. It is deterministic and lightweight.
    vector = [0.0] * max(16, int(dim))
    tokens = _tokenize_for_memory(text)
    if not tokens:
        return vector

    for token in tokens:
        idx = _stable_token_hash(token) % len(vector)
        vector[idx] += 1.0
        if len(token) > 5:
            idx2 = _stable_token_hash(token[:5]) % len(vector)
            vector[idx2] += 0.5
    return _normalize_vector(vector)


def _cosine_similarity(vec_a, vec_b):
    if not vec_a or not vec_b:
        return 0.0
    size = min(len(vec_a), len(vec_b))
    if size == 0:
        return 0.0
    return float(sum(float(vec_a[i]) * float(vec_b[i]) for i in range(size)))


def _memory_file_signature(path_obj: Path):
    try:
        stat = path_obj.stat()
        return f"{stat.st_size}:{getattr(stat, 'st_mtime_ns', int(stat.st_mtime * 1e9))}"
    except Exception:
        return "0:0"


def _load_memory_vector_index():
    default_index = {
        "version": 1,
        "backend": "hash-embedding-v1",
        "dimension": MEMORY_VECTOR_DIM,
        "updated_at": "",
        "items": [],
    }
    try:
        if not MEMORY_VECTOR_INDEX_FILE.exists():
            return default_index
        data = json.loads(MEMORY_VECTOR_INDEX_FILE.read_text(encoding="utf-8", errors="replace"))
        if not isinstance(data, dict):
            return default_index
        if not isinstance(data.get("items"), list):
            data["items"] = []
        return data
    except Exception:
        return default_index


def _save_memory_vector_index(index_data):
    _ensure_memory_paths()
    index_data["updated_at"] = datetime.now().isoformat(timespec="seconds")
    MEMORY_VECTOR_INDEX_FILE.write_text(
        json.dumps(index_data, ensure_ascii=True),
        encoding="utf-8",
        errors="replace",
    )


def _build_memory_vector_index(candidates, force_rebuild: bool = False, max_chunk_chars: int = 1600):
    index_data = _load_memory_vector_index()
    existing = {}
    if not force_rebuild:
        for item in index_data.get("items", []):
            if not isinstance(item, dict):
                continue
            item_id = str(item.get("id", "")).strip()
            if item_id:
                existing[item_id] = item

    rebuilt_items = []
    touched_ids = set()

    for path_obj in candidates:
        if not path_obj.exists():
            continue
        rel = path_obj.relative_to(WORKSPACE_ROOT).as_posix()
        signature = _memory_file_signature(path_obj)
        recency = _memory_file_recency_boost(path_obj)
        raw = path_obj.read_text(encoding="utf-8", errors="replace")
        chunks = _memory_chunks_from_text(raw, max_chunk_chars=max_chunk_chars)

        for idx, chunk in enumerate(chunks, start=1):
            chunk_hash = hashlib.sha1(chunk.encode("utf-8", errors="ignore")).hexdigest()
            item_id = f"{rel}|{idx}|{chunk_hash[:16]}"
            touched_ids.add(item_id)

            cached = existing.get(item_id)
            if cached and isinstance(cached.get("vector"), list):
                vector = cached["vector"]
            else:
                vector = _hash_embed_text(chunk, dim=MEMORY_VECTOR_DIM)

            rebuilt_items.append(
                {
                    "id": item_id,
                    "file": rel,
                    "chunk_index": idx,
                    "chunk_hash": chunk_hash,
                    "file_signature": signature,
                    "recency_boost": float(recency),
                    "snippet": chunk[:1200],
                    "vector": vector,
                }
            )

    index_data.update(
        {
            "version": 1,
            "backend": "hash-embedding-v1",
            "dimension": MEMORY_VECTOR_DIM,
            "items": rebuilt_items,
            "touched": len(touched_ids),
        }
    )
    _save_memory_vector_index(index_data)
    return index_data


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


def _serialize_call_result(result) -> str:
    text_chunks = []
    if hasattr(result, "content") and result.content:
        for item in result.content:
            chunk = getattr(item, "text", None)
            if chunk is not None:
                text_chunks.append(str(chunk))
    if text_chunks:
        return "\n".join(text_chunks)
    if getattr(result, "structuredContent", None) is not None:
        try:
            return json.dumps(result.structuredContent, ensure_ascii=True)
        except Exception:
            return str(result.structuredContent)
    return str(result)


def _parse_tabs_text(text: str):
    tabs = []
    for line in str(text).splitlines():
        match = TAB_LINE_RE.match(line.strip())
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


def _host(url: str) -> str:
    try:
        return (urlparse(url).hostname or "").lower()
    except Exception:
        return ""


def _hosts_match(expected_url: str, actual_url: str) -> bool:
    expected_host = _host(expected_url)
    actual_host = _host(actual_url)
    if not expected_host:
        return bool(actual_host)
    if expected_host == actual_host:
        return True
    return actual_host.endswith(f".{expected_host}")


# Define the tools available to the OpenAI model (Starting with the local ones)
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
                    "allow_dangerous": {"type": "boolean", "description": "Set true to bypass command safety blocklist.", "default": False},
                },
                "required": ["command"],
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

AVAILABLE_FUNCTIONS = {
    "calculate": calculate,
}

# MCP Client Setup
mcp_session = None
_mcp_exit_stack = None


async def _call_mcp_tool_raw(tool_name: str, args=None):
    if mcp_session is None:
        return {
            "ok": False,
            "tool": tool_name,
            "args": args or {},
            "error": "MCP session is not connected.",
            "text": "",
        }

    payload = args or {}
    try:
        result = await mcp_session.call_tool(tool_name, arguments=payload)
        is_error = bool(getattr(result, "isError", False))
        text = _serialize_call_result(result)
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


async def _list_tabs_state():
    call = await _call_mcp_tool_raw("browser_tabs", {"action": "list"})
    if not call["ok"]:
        return {"ok": False, "error": call["error"], "tabs": [], "current": None, "raw": call.get("text", "")}
    tabs = _parse_tabs_text(call.get("text", ""))
    current = next((tab for tab in tabs if tab["current"]), None)
    return {"ok": True, "error": None, "tabs": tabs, "current": current, "raw": call.get("text", "")}


async def _close_tab(index: int):
    return await _call_mcp_tool_raw("browser_tabs", {"action": "close", "index": index})


async def _select_tab(index: int):
    return await _call_mcp_tool_raw("browser_tabs", {"action": "select", "index": index})


async def _cleanup_blank_tabs(keep_current_blank=False):
    before = await _list_tabs_state()
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
        await _close_tab(tab["index"])
        tabs = [entry for entry in tabs if entry["index"] != tab["index"]]

    return await _list_tabs_state()


async def _ensure_owned_working_tab(preserve_current_blank=False):
    # Keep one active real tab and remove extra about:blank tabs to avoid model confusion.
    tabs_state = await _cleanup_blank_tabs(keep_current_blank=preserve_current_blank)
    if not tabs_state["ok"]:
        return tabs_state

    tabs = tabs_state["tabs"]
    if not tabs:
        await _call_mcp_tool_raw("browser_tabs", {"action": "new"})
        tabs_state = await _list_tabs_state()
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
        await _select_tab(target["index"])
        tabs_state = await _list_tabs_state()
    return tabs_state


async def _capture_page_state(include_snapshot=False):
    tabs_state = await _list_tabs_state()
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
        snap = await _call_mcp_tool_raw("browser_snapshot", {})
        if snap["ok"]:
            state["snapshot_hash"] = hashlib.sha1(snap.get("text", "").encode("utf-8", errors="ignore")).hexdigest()
    return state


async def _verify_step(tool_name: str, args: dict, before_state: dict, call_outcome: dict):
    if not call_outcome["ok"]:
        return {"ok": False, "reason": call_outcome.get("error", "Tool failed"), "details": {}}

    if tool_name == "browser_navigate":
        after = await _capture_page_state(include_snapshot=False)
        actual_url = after.get("url", "")
        expected_url = str(args.get("url", "")).strip()
        ok = bool(actual_url and actual_url != "about:blank" and _hosts_match(expected_url, actual_url))
        reason = (
            f"Expected host from '{expected_url}', current url is '{actual_url}'."
            if not ok
            else f"Navigation verified on '{actual_url}'."
        )
        return {"ok": ok, "reason": reason, "details": {"before": before_state, "after": after}}

    if tool_name == "browser_tabs":
        after = await _list_tabs_state()
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

    if tool_name in STATE_CHANGE_TOOLS:
        after = await _capture_page_state(include_snapshot=True)
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

    after = await _capture_page_state(include_snapshot=False)
    ok = bool(after.get("url") or tool_name in {"browser_close", "browser_install"})
    reason = "Tool succeeded and browser context is reachable." if ok else "Browser context could not be verified."
    return {"ok": ok, "reason": reason, "details": {"before": before_state, "after": after}}


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


async def browser_tabs_list(_kwargs_dict=None):
    state = await _list_tabs_state()
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


async def browser_tab_select(kwargs_dict):
    if mcp_session is None:
        return json.dumps({"status": "failed", "error": "MCP session is not connected."}, ensure_ascii=True)

    kwargs = kwargs_dict or {}
    target_index = kwargs.get("index")
    url_contains = str(kwargs.get("url_contains", "")).strip().lower()
    title_contains = str(kwargs.get("title_contains", "")).strip().lower()

    tabs_state = await _list_tabs_state()
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

    selected = await _select_tab(target["index"])
    if not selected["ok"]:
        return json.dumps({"status": "failed", "error": selected["error"]}, ensure_ascii=True)

    final_state = await _list_tabs_state()
    return json.dumps(
        {"status": "ok", "selected_index": target["index"], "current": final_state.get("current"), "tabs": final_state.get("tabs", [])},
        ensure_ascii=True,
    )


async def browser_close_blank_tabs(_kwargs_dict=None):
    state = await _cleanup_blank_tabs()
    if not state["ok"]:
        return json.dumps({"status": "failed", "error": state["error"]}, ensure_ascii=True)
    return json.dumps({"status": "ok", "tabs": state["tabs"], "current": state["current"]}, ensure_ascii=True)


async def fs_list(kwargs_dict=None):
    kwargs = kwargs_dict or {}
    path_value = kwargs.get("path", ".")
    recursive = bool(kwargs.get("recursive", False))
    include_hidden = bool(kwargs.get("include_hidden", False))
    max_entries = int(kwargs.get("max_entries", 200))
    max_entries = max(1, min(max_entries, 2000))

    try:
        root = _resolve_workspace_path(path_value, must_exist=True)
        if not root.is_dir():
            return json.dumps({"status": "failed", "error": f"Not a directory: {root}"}, ensure_ascii=True)

        iterator = root.rglob("*") if recursive else root.iterdir()
        entries = []
        for item in iterator:
            rel = item.relative_to(WORKSPACE_ROOT).as_posix()
            if not include_hidden and any(part.startswith(".") for part in Path(rel).parts):
                continue
            info = {
                "path": rel,
                "type": "dir" if item.is_dir() else "file",
            }
            if item.is_file():
                try:
                    info["size"] = item.stat().st_size
                except Exception:
                    info["size"] = None
            entries.append(info)
            if len(entries) >= max_entries:
                break

        entries.sort(key=lambda x: x["path"])
        return json.dumps(
            {
                "status": "ok",
                "root": root.relative_to(WORKSPACE_ROOT).as_posix() if root != WORKSPACE_ROOT else ".",
                "count": len(entries),
                "entries": entries,
            },
            ensure_ascii=True,
        )
    except Exception as e:
        return json.dumps({"status": "failed", "error": str(e)}, ensure_ascii=True)


async def fs_read(kwargs_dict):
    kwargs = kwargs_dict or {}
    path_value = kwargs.get("path")
    encoding = kwargs.get("encoding", "utf-8")
    max_chars = int(kwargs.get("max_chars", 20000))
    max_chars = max(200, min(max_chars, 500000))

    try:
        file_path = _resolve_workspace_path(path_value, must_exist=True)
        if not file_path.is_file():
            return json.dumps({"status": "failed", "error": f"Not a file: {file_path}"}, ensure_ascii=True)

        content = file_path.read_text(encoding=encoding, errors="replace")
        truncated = len(content) > max_chars
        payload = content[:max_chars]
        return json.dumps(
            {
                "status": "ok",
                "path": file_path.relative_to(WORKSPACE_ROOT).as_posix(),
                "chars": len(content),
                "truncated": truncated,
                "content": payload,
            },
            ensure_ascii=True,
        )
    except Exception as e:
        return json.dumps({"status": "failed", "error": str(e)}, ensure_ascii=True)


async def fs_read_batch(kwargs_dict):
    kwargs = kwargs_dict or {}
    paths = kwargs.get("paths", [])
    encoding = kwargs.get("encoding", "utf-8")
    missing_ok = bool(kwargs.get("missing_ok", True))
    max_chars_per_file = int(kwargs.get("max_chars_per_file", 12000))
    max_chars_per_file = max(200, min(max_chars_per_file, 500000))

    if not isinstance(paths, list) or not paths:
        return json.dumps({"status": "failed", "error": "paths must be a non-empty array"}, ensure_ascii=True)

    results = []
    failures = []
    for raw_path in paths[:200]:
        try:
            file_path = _resolve_workspace_path(raw_path, must_exist=True)
            if not file_path.is_file():
                results.append({"path": str(raw_path), "status": "failed", "error": "Not a file"})
                failures.append(str(raw_path))
                continue
            content = file_path.read_text(encoding=encoding, errors="replace")
            rel_path = file_path.relative_to(WORKSPACE_ROOT).as_posix()
            results.append(
                {
                    "path": rel_path,
                    "status": "ok",
                    "chars": len(content),
                    "truncated": len(content) > max_chars_per_file,
                    "content": content[:max_chars_per_file],
                }
            )
        except Exception as e:
            if missing_ok:
                results.append({"path": str(raw_path), "status": "failed", "error": str(e)})
            else:
                return json.dumps({"status": "failed", "error": str(e), "path": str(raw_path)}, ensure_ascii=True)
            failures.append(str(raw_path))

    return json.dumps(
        {
            "status": "ok",
            "count": len(results),
            "failed_count": len(failures),
            "results": results,
        },
        ensure_ascii=True,
    )


async def fs_edit_lines(kwargs_dict):
    kwargs = kwargs_dict or {}
    path_value = kwargs.get("path")
    start_line = int(kwargs.get("start_line", 0))
    end_line = int(kwargs.get("end_line", 0))
    replacement = str(kwargs.get("replacement", ""))
    encoding = kwargs.get("encoding", "utf-8")
    strict = bool(kwargs.get("strict", True))
    dry_run = bool(kwargs.get("dry_run", False))

    try:
        if start_line < 1 or end_line < 1 or end_line < start_line:
            return json.dumps({"status": "failed", "error": "Invalid line range"}, ensure_ascii=True)

        file_path = _resolve_workspace_path(path_value, must_exist=True)
        if not file_path.is_file():
            return json.dumps({"status": "failed", "error": f"Not a file: {file_path}"}, ensure_ascii=True)

        original = file_path.read_text(encoding=encoding, errors="replace")
        newline = "\r\n" if "\r\n" in original else "\n"
        original_lines = original.splitlines(keepends=True)
        line_count = len(original_lines)

        if strict and end_line > line_count:
            return json.dumps(
                {"status": "failed", "error": f"Line range exceeds file length {line_count}"},
                ensure_ascii=True,
            )

        safe_end = min(end_line, line_count)
        if start_line > line_count + 1:
            return json.dumps(
                {"status": "failed", "error": f"start_line {start_line} exceeds file length {line_count}"},
                ensure_ascii=True,
            )

        replacement_block = replacement.replace("\r\n", "\n").replace("\r", "\n")
        replacement_lines = replacement_block.split("\n")
        replacement_with_endings = []
        for idx, chunk in enumerate(replacement_lines):
            is_last = idx == len(replacement_lines) - 1
            preserve_line_break = safe_end < line_count
            if is_last and not replacement.endswith(("\n", "\r\n", "\r")) and not preserve_line_break:
                replacement_with_endings.append(chunk)
            else:
                replacement_with_endings.append(chunk + newline)
        if replacement == "":
            replacement_with_endings = []

        updated_lines = original_lines[: start_line - 1] + replacement_with_endings + original_lines[safe_end:]
        updated = "".join(updated_lines)

        changed = updated != original
        wrote = False
        if changed and not dry_run:
            file_path.write_text(updated, encoding=encoding, errors="replace")
            wrote = True

        return json.dumps(
            {
                "status": "ok",
                "path": file_path.relative_to(WORKSPACE_ROOT).as_posix(),
                "line_count_before": line_count,
                "line_count_after": len(updated.splitlines()),
                "changed": changed,
                "wrote": wrote,
                "dry_run": dry_run,
                "range": {"start_line": start_line, "end_line": end_line, "effective_end_line": safe_end},
                "before_hash": hashlib.sha1(original.encode("utf-8", errors="ignore")).hexdigest(),
                "after_hash": hashlib.sha1(updated.encode("utf-8", errors="ignore")).hexdigest(),
            },
            ensure_ascii=True,
        )
    except Exception as e:
        return json.dumps({"status": "failed", "error": str(e)}, ensure_ascii=True)


async def fs_insert_lines(kwargs_dict):
    kwargs = kwargs_dict or {}
    path_value = kwargs.get("path")
    line = int(kwargs.get("line", 0))
    content = str(kwargs.get("content", ""))
    encoding = kwargs.get("encoding", "utf-8")
    dry_run = bool(kwargs.get("dry_run", False))

    try:
        if line < 1:
            return json.dumps({"status": "failed", "error": "line must be >= 1"}, ensure_ascii=True)

        file_path = _resolve_workspace_path(path_value, must_exist=True)
        if not file_path.is_file():
            return json.dumps({"status": "failed", "error": f"Not a file: {file_path}"}, ensure_ascii=True)

        original = file_path.read_text(encoding=encoding, errors="replace")
        newline = "\r\n" if "\r\n" in original else "\n"
        lines = original.splitlines(keepends=True)
        line_count = len(lines)
        insert_at = min(line - 1, line_count)

        normalized = content.replace("\r\n", "\n").replace("\r", "\n")
        block = normalized.replace("\n", newline)
        if block and not block.endswith(newline):
            block += newline

        updated = "".join(lines[:insert_at]) + block + "".join(lines[insert_at:])
        changed = updated != original
        wrote = False
        if changed and not dry_run:
            file_path.write_text(updated, encoding=encoding, errors="replace")
            wrote = True

        return json.dumps(
            {
                "status": "ok",
                "path": file_path.relative_to(WORKSPACE_ROOT).as_posix(),
                "insert_line": line,
                "effective_insert_index": insert_at + 1,
                "line_count_before": line_count,
                "line_count_after": len(updated.splitlines()),
                "changed": changed,
                "wrote": wrote,
                "dry_run": dry_run,
                "before_hash": hashlib.sha1(original.encode("utf-8", errors="ignore")).hexdigest(),
                "after_hash": hashlib.sha1(updated.encode("utf-8", errors="ignore")).hexdigest(),
            },
            ensure_ascii=True,
        )
    except Exception as e:
        return json.dumps({"status": "failed", "error": str(e)}, ensure_ascii=True)


async def fs_write(kwargs_dict):
    kwargs = kwargs_dict or {}
    path_value = kwargs.get("path")
    content = str(kwargs.get("content", ""))
    append = bool(kwargs.get("append", False))
    encoding = kwargs.get("encoding", "utf-8")

    try:
        file_path = _resolve_workspace_path(path_value, must_exist=False)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if append else "w"
        with file_path.open(mode, encoding=encoding, errors="replace", newline="") as fh:
            fh.write(content)
        return json.dumps(
            {
                "status": "ok",
                "path": file_path.relative_to(WORKSPACE_ROOT).as_posix(),
                "append": append,
                "written_chars": len(content),
            },
            ensure_ascii=True,
        )
    except Exception as e:
        return json.dumps({"status": "failed", "error": str(e)}, ensure_ascii=True)


async def fs_copy(kwargs_dict):
    kwargs = kwargs_dict or {}
    source_value = kwargs.get("source")
    destination_value = kwargs.get("destination")
    overwrite = bool(kwargs.get("overwrite", False))

    try:
        source_path = _resolve_workspace_path(source_value, must_exist=True)
        destination_path = _resolve_workspace_path(destination_value, must_exist=False)

        if source_path == destination_path:
            return json.dumps({"status": "failed", "error": "source and destination are the same path"}, ensure_ascii=True)

        if destination_path.exists():
            if not overwrite:
                return json.dumps(
                    {"status": "failed", "error": f"Destination already exists: {destination_path}"},
                    ensure_ascii=True,
                )
            if destination_path.is_dir():
                shutil.rmtree(destination_path)
            else:
                destination_path.unlink()

        destination_path.parent.mkdir(parents=True, exist_ok=True)
        if source_path.is_dir():
            shutil.copytree(source_path, destination_path)
            copied_type = "dir"
        else:
            shutil.copy2(source_path, destination_path)
            copied_type = "file"

        return json.dumps(
            {
                "status": "ok",
                "type": copied_type,
                "source": source_path.relative_to(WORKSPACE_ROOT).as_posix(),
                "destination": destination_path.relative_to(WORKSPACE_ROOT).as_posix(),
                "overwrite": overwrite,
            },
            ensure_ascii=True,
        )
    except Exception as e:
        return json.dumps({"status": "failed", "error": str(e)}, ensure_ascii=True)


async def fs_move(kwargs_dict):
    kwargs = kwargs_dict or {}
    source_value = kwargs.get("source")
    destination_value = kwargs.get("destination")
    overwrite = bool(kwargs.get("overwrite", False))

    try:
        source_path = _resolve_workspace_path(source_value, must_exist=True)
        destination_path = _resolve_workspace_path(destination_value, must_exist=False)

        if source_path == destination_path:
            return json.dumps({"status": "ok", "source": str(source_path), "destination": str(destination_path), "noop": True}, ensure_ascii=True)

        if destination_path.exists():
            if not overwrite:
                return json.dumps(
                    {"status": "failed", "error": f"Destination already exists: {destination_path}"},
                    ensure_ascii=True,
                )
            if destination_path.is_dir():
                shutil.rmtree(destination_path)
            else:
                destination_path.unlink()

        destination_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source_path), str(destination_path))

        return json.dumps(
            {
                "status": "ok",
                "source": source_path.relative_to(WORKSPACE_ROOT).as_posix(),
                "destination": destination_path.relative_to(WORKSPACE_ROOT).as_posix(),
                "overwrite": overwrite,
            },
            ensure_ascii=True,
        )
    except Exception as e:
        return json.dumps({"status": "failed", "error": str(e)}, ensure_ascii=True)


async def fs_delete(kwargs_dict):
    kwargs = kwargs_dict or {}
    path_value = kwargs.get("path")
    recursive = bool(kwargs.get("recursive", False))
    missing_ok = bool(kwargs.get("missing_ok", False))

    try:
        target_path = _resolve_workspace_path(path_value, must_exist=False)

        if not target_path.exists():
            if missing_ok:
                return json.dumps(
                    {
                        "status": "ok",
                        "path": target_path.relative_to(WORKSPACE_ROOT).as_posix(),
                        "deleted": False,
                        "missing": True,
                    },
                    ensure_ascii=True,
                )
            return json.dumps({"status": "failed", "error": f"Path does not exist: {target_path}"}, ensure_ascii=True)

        if target_path.is_dir():
            if recursive:
                shutil.rmtree(target_path)
            else:
                try:
                    target_path.rmdir()
                except OSError:
                    return json.dumps(
                        {"status": "failed", "error": "Directory is not empty. Set recursive=true to delete it."},
                        ensure_ascii=True,
                    )
            deleted_type = "dir"
        else:
            target_path.unlink()
            deleted_type = "file"

        return json.dumps(
            {
                "status": "ok",
                "path": target_path.relative_to(WORKSPACE_ROOT).as_posix(),
                "deleted": True,
                "type": deleted_type,
                "recursive": recursive,
            },
            ensure_ascii=True,
        )
    except Exception as e:
        return json.dumps({"status": "failed", "error": str(e)}, ensure_ascii=True)


async def fs_patch(kwargs_dict):
    kwargs = kwargs_dict or {}
    path_value = kwargs.get("path")
    edits = kwargs.get("edits", [])
    encoding = kwargs.get("encoding", "utf-8")
    strict = bool(kwargs.get("strict", True))
    create_if_missing = bool(kwargs.get("create_if_missing", False))
    dry_run = bool(kwargs.get("dry_run", False))

    try:
        if not isinstance(edits, list) or len(edits) == 0:
            return json.dumps({"status": "failed", "error": "edits must be a non-empty array"}, ensure_ascii=True)

        file_path = _resolve_workspace_path(path_value, must_exist=False)
        exists = file_path.exists()
        if not exists and not create_if_missing:
            return json.dumps({"status": "failed", "error": f"File does not exist: {file_path}"}, ensure_ascii=True)

        original = ""
        if exists:
            if not file_path.is_file():
                return json.dumps({"status": "failed", "error": f"Not a file: {file_path}"}, ensure_ascii=True)
            original = file_path.read_text(encoding=encoding, errors="replace")

        updated = original
        edit_results = []
        missed = []

        for idx, edit in enumerate(edits, start=1):
            if not isinstance(edit, dict):
                return json.dumps({"status": "failed", "error": f"edit #{idx} must be an object"}, ensure_ascii=True)
            find_value = edit.get("find")
            replace_value = str(edit.get("replace", ""))
            use_regex = bool(edit.get("regex", False))
            count_raw = int(edit.get("count", 0))

            if find_value is None or find_value == "":
                return json.dumps({"status": "failed", "error": f"edit #{idx} missing non-empty 'find'"}, ensure_ascii=True)

            try:
                if use_regex:
                    limit = count_raw if count_raw > 0 else 0
                    next_text, applied = re.subn(str(find_value), replace_value, updated, count=limit, flags=re.MULTILINE)
                else:
                    find_text = str(find_value)
                    total_matches = updated.count(find_text)
                    if count_raw > 0:
                        applied = min(total_matches, count_raw)
                        next_text = updated.replace(find_text, replace_value, count_raw)
                    else:
                        applied = total_matches
                        next_text = updated.replace(find_text, replace_value)
            except re.error as rex:
                return json.dumps({"status": "failed", "error": f"Regex error in edit #{idx}: {rex}"}, ensure_ascii=True)

            edit_results.append(
                {
                    "index": idx,
                    "matches": applied,
                    "regex": use_regex,
                    "count": count_raw,
                }
            )
            if applied == 0:
                missed.append(idx)
            updated = next_text

        changed = updated != original
        if strict and missed:
            return json.dumps(
                {
                    "status": "failed",
                    "error": "One or more edits had zero matches in strict mode.",
                    "missed_edits": missed,
                    "changed": changed,
                    "dry_run": dry_run,
                    "path": file_path.relative_to(WORKSPACE_ROOT).as_posix(),
                    "edit_results": edit_results,
                },
                ensure_ascii=True,
            )

        wrote = False
        if not dry_run:
            if changed or (not exists and create_if_missing):
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.write_text(updated, encoding=encoding, errors="replace")
                wrote = True

        before_hash = hashlib.sha1(original.encode("utf-8", errors="ignore")).hexdigest()
        after_hash = hashlib.sha1(updated.encode("utf-8", errors="ignore")).hexdigest()

        return json.dumps(
            {
                "status": "ok",
                "path": file_path.relative_to(WORKSPACE_ROOT).as_posix(),
                "changed": changed,
                "wrote": wrote,
                "dry_run": dry_run,
                "strict": strict,
                "missed_edits": missed,
                "edit_results": edit_results,
                "before_hash": before_hash,
                "after_hash": after_hash,
                "chars_before": len(original),
                "chars_after": len(updated),
            },
            ensure_ascii=True,
        )
    except Exception as e:
        return json.dumps({"status": "failed", "error": str(e)}, ensure_ascii=True)


async def fs_search(kwargs_dict):
    kwargs = kwargs_dict or {}
    pattern = kwargs.get("pattern")
    path_value = kwargs.get("path", ".")
    file_glob = kwargs.get("file_glob", "*")
    case_sensitive = bool(kwargs.get("case_sensitive", False))
    use_regex = bool(kwargs.get("regex", False))
    max_results = int(kwargs.get("max_results", 200))
    max_results = max(1, min(max_results, 2000))

    try:
        if not pattern:
            return json.dumps({"status": "failed", "error": "pattern is required"}, ensure_ascii=True)

        root = _resolve_workspace_path(path_value, must_exist=True)
        if not root.is_dir():
            return json.dumps({"status": "failed", "error": f"Not a directory: {root}"}, ensure_ascii=True)

        flags = 0 if case_sensitive else re.IGNORECASE
        compiled = re.compile(pattern if use_regex else re.escape(pattern), flags=flags)

        matches = []
        for file_path in root.rglob("*"):
            if not file_path.is_file():
                continue
            if not fnmatch.fnmatch(file_path.name, file_glob):
                continue

            try:
                text = file_path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue

            for line_number, line in enumerate(text.splitlines(), start=1):
                if compiled.search(line):
                    matches.append(
                        {
                            "path": file_path.relative_to(WORKSPACE_ROOT).as_posix(),
                            "line": line_number,
                            "text": line.strip()[:400],
                        }
                    )
                    if len(matches) >= max_results:
                        break
            if len(matches) >= max_results:
                break

        return json.dumps(
            {
                "status": "ok",
                "root": root.relative_to(WORKSPACE_ROOT).as_posix() if root != WORKSPACE_ROOT else ".",
                "count": len(matches),
                "matches": matches,
            },
            ensure_ascii=True,
        )
    except Exception as e:
        return json.dumps({"status": "failed", "error": str(e)}, ensure_ascii=True)


async def fs_analyze_file(kwargs_dict):
    kwargs = kwargs_dict or {}
    path_value = kwargs.get("path")
    encoding = kwargs.get("encoding", "utf-8")
    max_chars = int(kwargs.get("max_chars", 200000))
    include_preview = bool(kwargs.get("include_preview", True))
    max_chars = max(500, min(max_chars, 2000000))

    try:
        file_path = _resolve_workspace_path(path_value, must_exist=True)
        if not file_path.is_file():
            return json.dumps({"status": "failed", "error": f"Not a file: {file_path}"}, ensure_ascii=True)

        content = file_path.read_text(encoding=encoding, errors="replace")
        truncated = len(content) > max_chars
        parse_content = content[:max_chars]
        analysis = _analyze_content(file_path, parse_content)

        payload = {
            "status": "ok",
            "path": file_path.relative_to(WORKSPACE_ROOT).as_posix(),
            "chars": len(content),
            "truncated_for_analysis": truncated,
            "analysis": analysis,
        }
        if include_preview:
            payload["preview"] = parse_content[:1200]
        return json.dumps(payload, ensure_ascii=True)
    except Exception as e:
        return json.dumps({"status": "failed", "error": str(e)}, ensure_ascii=True)


async def codebase_analyze(kwargs_dict=None):
    kwargs = kwargs_dict or {}
    path_value = kwargs.get("path", ".")
    max_files = int(kwargs.get("max_files", 1200))
    include_hidden = bool(kwargs.get("include_hidden", False))
    top_n_large_files = int(kwargs.get("top_n_large_files", 20))
    max_files = max(10, min(max_files, 10000))
    top_n_large_files = max(3, min(top_n_large_files, 100))

    try:
        root = _resolve_workspace_path(path_value, must_exist=True)
        if not root.is_dir():
            return json.dumps({"status": "failed", "error": f"Not a directory: {root}"}, ensure_ascii=True)

        files = []
        dirs_set = set()
        lang_counts = {}
        total_size = 0

        for item in root.rglob("*"):
            rel = item.relative_to(WORKSPACE_ROOT).as_posix()
            if not include_hidden and any(part.startswith(".") for part in Path(rel).parts):
                continue
            if _is_noise_path(Path(rel)):
                continue

            if item.is_dir():
                dirs_set.add(rel)
                continue
            if not item.is_file():
                continue
            if not _is_probably_text_source(Path(rel)):
                continue

            try:
                size = item.stat().st_size
            except Exception:
                size = 0
            total_size += size
            language = _language_from_path(item)
            lang_counts[language] = lang_counts.get(language, 0) + 1
            files.append({"path": rel, "size": size, "language": language})

            if len(files) >= max_files:
                break

        key_file_names = {
            "README.md",
            "readme.md",
            "package.json",
            "requirements.txt",
            "pyproject.toml",
            "tsconfig.json",
            "Dockerfile",
            ".env",
        }
        key_files = [f["path"] for f in files if Path(f["path"]).name in key_file_names]
        largest_files = sorted(files, key=lambda x: x["size"], reverse=True)[:top_n_large_files]
        language_distribution = sorted(
            [{"language": lang, "files": count} for lang, count in lang_counts.items()],
            key=lambda x: x["files"],
            reverse=True,
        )

        return json.dumps(
            {
                "status": "ok",
                "root": root.relative_to(WORKSPACE_ROOT).as_posix() if root != WORKSPACE_ROOT else ".",
                "files_scanned": len(files),
                "directories_scanned": len(dirs_set),
                "total_size_bytes": total_size,
                "language_distribution": language_distribution,
                "key_files": key_files[:80],
                "largest_files": largest_files,
            },
            ensure_ascii=True,
        )
    except Exception as e:
        return json.dumps({"status": "failed", "error": str(e)}, ensure_ascii=True)


async def reasoning_plan(kwargs_dict):
    kwargs = kwargs_dict or {}
    goal = str(kwargs.get("goal", "")).strip()
    context = str(kwargs.get("context", "")).strip()
    max_steps = int(kwargs.get("max_steps", 8))
    max_steps = max(3, min(max_steps, 20))

    if not goal:
        return json.dumps({"status": "failed", "error": "goal is required"}, ensure_ascii=True)

    separators = r"(?:\.\s+|;\s+|\n+| then | and then | after that )"
    pieces = [chunk.strip(" -\t\r\n") for chunk in re.split(separators, goal) if chunk.strip(" -\t\r\n")]
    if not pieces:
        pieces = [goal]

    steps = []
    for idx, piece in enumerate(pieces[:max_steps], start=1):
        steps.append({"step": idx, "action": piece})

    while len(steps) < min(3, max_steps):
        steps.append({"step": len(steps) + 1, "action": "Verify results and adjust based on observed output."})

    assumptions = []
    if context:
        assumptions.append("Provided context and constraints are accurate.")
    assumptions.extend(
        [
            "Required tools and permissions are available.",
            "Target paths and URLs are reachable.",
        ]
    )

    risks = [
        "Missing permissions or blocked resources can interrupt execution.",
        "Dynamic websites may require selector fallback and retries.",
    ]

    return json.dumps(
        {
            "status": "ok",
            "goal": goal,
            "context": context,
            "assumptions": assumptions,
            "plan_steps": steps[:max_steps],
            "risks": risks,
        },
        ensure_ascii=True,
    )


async def memory_log(kwargs_dict):
    kwargs = kwargs_dict or {}
    content = str(kwargs.get("content", "")).strip()
    role = str(kwargs.get("role", "event")).strip() or "event"
    importance = int(kwargs.get("importance", 3))
    tags = kwargs.get("tags", [])
    if not isinstance(tags, list):
        tags = []

    if not content:
        return json.dumps({"status": "failed", "error": "content is required"}, ensure_ascii=True)

    try:
        file_path = _append_daily_memory(content=content, role=role, importance=importance, tags=tags)
        return json.dumps(
            {
                "status": "ok",
                "file": file_path.relative_to(WORKSPACE_ROOT).as_posix(),
                "role": role,
                "importance": max(1, min(int(importance), 10)),
            },
            ensure_ascii=True,
        )
    except Exception as e:
        return json.dumps({"status": "failed", "error": str(e)}, ensure_ascii=True)


async def memory_promote(kwargs_dict):
    kwargs = kwargs_dict or {}
    fact = str(kwargs.get("fact", "")).strip()
    importance = int(kwargs.get("importance", 7))
    tags = kwargs.get("tags", [])
    if not isinstance(tags, list):
        tags = []
    if not fact:
        return json.dumps({"status": "failed", "error": "fact is required"}, ensure_ascii=True)

    try:
        _ensure_memory_paths()
        now = datetime.now().isoformat(timespec="seconds")
        safe_importance = max(1, min(int(importance), 10))
        tag_text = ",".join(sorted({str(tag).strip() for tag in tags if str(tag).strip()}))
        entry = f"- [{now}] (importance:{safe_importance}) {fact}"
        if tag_text:
            entry += f" [tags:{tag_text}]"
        entry += "\n"

        with LONG_TERM_MEMORY_FILE.open("a", encoding="utf-8", errors="replace", newline="") as fh:
            fh.write(entry)

        return json.dumps(
            {
                "status": "ok",
                "file": LONG_TERM_MEMORY_FILE.relative_to(WORKSPACE_ROOT).as_posix(),
                "importance": safe_importance,
            },
            ensure_ascii=True,
        )
    except Exception as e:
        return json.dumps({"status": "failed", "error": str(e)}, ensure_ascii=True)


async def memory_get(kwargs_dict=None):
    kwargs = kwargs_dict or {}
    date_str = str(kwargs.get("date", "")).strip()
    file_value = str(kwargs.get("file", "")).strip()
    start_line = int(kwargs.get("start_line", 1))
    end_line = int(kwargs.get("end_line", 200))
    max_chars = int(kwargs.get("max_chars", 20000))
    max_chars = max(200, min(max_chars, 500000))

    try:
        if start_line < 1:
            start_line = 1
        if end_line < start_line:
            end_line = start_line

        if file_value:
            path_obj = _resolve_workspace_path(file_value, must_exist=True)
        elif date_str:
            target_date = datetime.strptime(date_str, "%Y-%m-%d")
            path_obj = _daily_memory_file(target_date)
            if not path_obj.exists():
                return json.dumps({"status": "failed", "error": f"No daily memory for {date_str}"}, ensure_ascii=True)
        else:
            path_obj = LONG_TERM_MEMORY_FILE
            if not path_obj.exists():
                _ensure_memory_paths()

        text, lines = _read_file_lines(path_obj)
        total_lines = len(lines)
        safe_end = min(end_line, total_lines)
        snippet = "\n".join(lines[start_line - 1:safe_end])
        snippet = snippet[:max_chars]
        return json.dumps(
            {
                "status": "ok",
                "file": path_obj.relative_to(WORKSPACE_ROOT).as_posix(),
                "start_line": start_line,
                "end_line": safe_end,
                "total_lines": total_lines,
                "content": snippet,
                "truncated": len(snippet) >= max_chars,
            },
            ensure_ascii=True,
        )
    except Exception as e:
        return json.dumps({"status": "failed", "error": str(e)}, ensure_ascii=True)


async def memory_search(kwargs_dict):
    kwargs = kwargs_dict or {}
    query = str(kwargs.get("query", "")).strip()
    top_k = int(kwargs.get("top_k", 6))
    include_long_term = bool(kwargs.get("include_long_term", True))
    days_back = int(kwargs.get("days_back", 14))
    use_semantic = bool(kwargs.get("use_semantic", True))
    rebuild_index = bool(kwargs.get("rebuild_index", False))
    semantic_weight = float(kwargs.get("semantic_weight", 0.65))
    lexical_weight = float(kwargs.get("lexical_weight", 0.35))
    max_chunk_chars = int(kwargs.get("max_chunk_chars", 1600))
    top_k = max(1, min(top_k, 30))
    days_back = max(1, min(days_back, 120))
    max_chunk_chars = max(400, min(max_chunk_chars, 5000))
    semantic_weight = max(0.0, min(semantic_weight, 1.0))
    lexical_weight = max(0.0, min(lexical_weight, 1.0))
    if semantic_weight == 0.0 and lexical_weight == 0.0:
        lexical_weight = 1.0
    weight_sum = semantic_weight + lexical_weight
    semantic_weight = semantic_weight / weight_sum
    lexical_weight = lexical_weight / weight_sum

    if not query:
        return json.dumps({"status": "failed", "error": "query is required"}, ensure_ascii=True)

    try:
        _ensure_memory_paths()
        candidates = []
        today = datetime.now()
        for offset in range(days_back):
            day = today - timedelta(days=offset)
            file_path = _daily_memory_file(day)
            if file_path.exists():
                candidates.append(file_path)
        if include_long_term and LONG_TERM_MEMORY_FILE.exists():
            candidates.append(LONG_TERM_MEMORY_FILE)

        if not candidates:
            return json.dumps(
                {
                    "status": "ok",
                    "query": query,
                    "count": 0,
                    "results": [],
                    "semantic_enabled": use_semantic,
                },
                ensure_ascii=True,
            )

        query_tokens = _tokenize_for_memory(query)
        lexical_by_id = {}
        for path_obj in candidates:
            raw = path_obj.read_text(encoding="utf-8", errors="replace")
            chunks = _memory_chunks_from_text(raw, max_chunk_chars=max_chunk_chars)
            recency = _memory_file_recency_boost(path_obj)
            file_rel = path_obj.relative_to(WORKSPACE_ROOT).as_posix()
            for idx, chunk in enumerate(chunks, start=1):
                chunk_hash = hashlib.sha1(chunk.encode("utf-8", errors="ignore")).hexdigest()
                chunk_id = f"{file_rel}|{idx}|{chunk_hash[:16]}"
                lexical = _memory_chunk_score(query_tokens, chunk, recency_boost=recency)
                lexical_by_id[chunk_id] = {
                    "file": file_rel,
                    "chunk_index": idx,
                    "chunk_hash": chunk_hash,
                    "snippet": chunk[:1200],
                    "lexical_score_raw": float(max(0.0, lexical)),
                    "recency_boost": float(recency),
                }

        semantic_by_id = {}
        index_backend = "none"
        index_items = 0
        if use_semantic:
            index_data = _build_memory_vector_index(
                candidates=candidates,
                force_rebuild=rebuild_index,
                max_chunk_chars=max_chunk_chars,
            )
            index_backend = str(index_data.get("backend", "hash-embedding-v1"))
            index_items = len(index_data.get("items", []))
            query_vector = _hash_embed_text(query, dim=int(index_data.get("dimension", MEMORY_VECTOR_DIM)))
            for item in index_data.get("items", []):
                if not isinstance(item, dict):
                    continue
                item_id = str(item.get("id", "")).strip()
                if not item_id:
                    continue
                semantic = _cosine_similarity(query_vector, item.get("vector", []))
                semantic = max(0.0, float(semantic))
                semantic_by_id[item_id] = {
                    "semantic_score_raw": semantic,
                    "file": item.get("file", ""),
                    "chunk_index": int(item.get("chunk_index", 0) or 0),
                    "snippet": str(item.get("snippet", ""))[:1200],
                    "recency_boost": float(item.get("recency_boost", 0.0)),
                }

        max_lexical = max((row.get("lexical_score_raw", 0.0) for row in lexical_by_id.values()), default=0.0)
        all_ids = set(lexical_by_id.keys()) | set(semantic_by_id.keys())
        ranked = []
        for chunk_id in all_ids:
            lexical_row = lexical_by_id.get(chunk_id, {})
            semantic_row = semantic_by_id.get(chunk_id, {})
            lexical_raw = float(lexical_row.get("lexical_score_raw", 0.0))
            lexical_norm = lexical_raw / max_lexical if max_lexical > 0 else 0.0
            semantic_raw = float(semantic_row.get("semantic_score_raw", 0.0))
            recency = float(lexical_row.get("recency_boost", semantic_row.get("recency_boost", 0.0)))
            combined = (lexical_weight * lexical_norm) + (semantic_weight * semantic_raw) + (0.03 * recency)

            if combined <= 0:
                continue

            file_rel = lexical_row.get("file") or semantic_row.get("file") or ""
            chunk_index = int(lexical_row.get("chunk_index") or semantic_row.get("chunk_index") or 0)
            snippet = lexical_row.get("snippet") or semantic_row.get("snippet") or ""
            ranked.append(
                {
                    "score": combined,
                    "combined_score": combined,
                    "lexical_score": lexical_norm,
                    "semantic_score": semantic_raw,
                    "file": file_rel,
                    "chunk_index": chunk_index,
                    "snippet": snippet[:1200],
                }
            )

        ranked.sort(key=lambda item: item["score"], reverse=True)
        results = ranked[:top_k]
        return json.dumps(
            {
                "status": "ok",
                "query": query,
                "count": len(results),
                "semantic_enabled": use_semantic,
                "weights": {"lexical": lexical_weight, "semantic": semantic_weight},
                "vector_index": {
                    "backend": index_backend,
                    "file": MEMORY_VECTOR_INDEX_FILE.relative_to(WORKSPACE_ROOT).as_posix(),
                    "items": index_items,
                },
                "results": results,
            },
            ensure_ascii=True,
        )
    except Exception as e:
        return json.dumps({"status": "failed", "error": str(e)}, ensure_ascii=True)


async def memory_bootstrap(kwargs_dict=None):
    kwargs = kwargs_dict or {}
    include_long_term = bool(kwargs.get("include_long_term", True))
    max_chars = int(kwargs.get("max_chars", 24000))
    max_chars = max(1000, min(max_chars, 200000))

    try:
        _ensure_memory_paths()
        today = datetime.now()
        yesterday = today - timedelta(days=1)
        files = []
        today_file = _daily_memory_file(today)
        yday_file = _daily_memory_file(yesterday)
        if today_file.exists():
            files.append(today_file)
        if yday_file.exists():
            files.append(yday_file)
        if include_long_term and LONG_TERM_MEMORY_FILE.exists():
            files.append(LONG_TERM_MEMORY_FILE)

        parts = []
        for file_path in files:
            text = file_path.read_text(encoding="utf-8", errors="replace")
            rel = file_path.relative_to(WORKSPACE_ROOT).as_posix()
            parts.append(f"### {rel}\n{text.strip()}\n")

        merged = "\n\n".join(parts).strip()[:max_chars]
        return json.dumps(
            {
                "status": "ok",
                "files": [file_path.relative_to(WORKSPACE_ROOT).as_posix() for file_path in files],
                "content": merged,
                "truncated": len(merged) >= max_chars,
            },
            ensure_ascii=True,
        )
    except Exception as e:
        return json.dumps({"status": "failed", "error": str(e)}, ensure_ascii=True)


async def memory_reindex(kwargs_dict=None):
    kwargs = kwargs_dict or {}
    include_long_term = bool(kwargs.get("include_long_term", True))
    days_back = int(kwargs.get("days_back", 30))
    max_chunk_chars = int(kwargs.get("max_chunk_chars", 1600))
    force_rebuild = bool(kwargs.get("force_rebuild", True))
    days_back = max(1, min(days_back, 365))
    max_chunk_chars = max(400, min(max_chunk_chars, 5000))

    try:
        _ensure_memory_paths()
        candidates = []
        today = datetime.now()
        for offset in range(days_back):
            day = today - timedelta(days=offset)
            file_path = _daily_memory_file(day)
            if file_path.exists():
                candidates.append(file_path)
        if include_long_term and LONG_TERM_MEMORY_FILE.exists():
            candidates.append(LONG_TERM_MEMORY_FILE)

        index_data = _build_memory_vector_index(
            candidates=candidates,
            force_rebuild=force_rebuild,
            max_chunk_chars=max_chunk_chars,
        )
        return json.dumps(
            {
                "status": "ok",
                "files_indexed": len(candidates),
                "vector_index_file": MEMORY_VECTOR_INDEX_FILE.relative_to(WORKSPACE_ROOT).as_posix(),
                "vector_backend": index_data.get("backend", "hash-embedding-v1"),
                "dimension": int(index_data.get("dimension", MEMORY_VECTOR_DIM)),
                "items": len(index_data.get("items", [])),
                "updated_at": index_data.get("updated_at", ""),
            },
            ensure_ascii=True,
        )
    except Exception as e:
        return json.dumps({"status": "failed", "error": str(e)}, ensure_ascii=True)


async def tool_catalog(kwargs_dict=None):
    kwargs = kwargs_dict or {}
    only_callable = bool(kwargs.get("only_callable", False))

    schema_map = {}
    for tool in AGENT_TOOLS:
        fn = tool.get("function", {})
        name = fn.get("name")
        if name:
            schema_map[name] = {
                "description": fn.get("description", ""),
                "parameters": fn.get("parameters", {"type": "object", "properties": {}, "required": []}),
            }

    tools = []
    for name, meta in sorted(schema_map.items(), key=lambda item: item[0]):
        callable_now = name in AVAILABLE_FUNCTIONS
        if only_callable and not callable_now:
            continue
        tools.append(
            {
                "name": name,
                "callable": callable_now,
                "description": meta["description"],
                "parameters": meta["parameters"],
            }
        )

    return json.dumps(
        {
            "status": "ok",
            "count": len(tools),
            "tools": tools,
        },
        ensure_ascii=True,
    )


async def _invoke_tool_by_name(tool_name: str, arguments: dict):
    target = AVAILABLE_FUNCTIONS.get(tool_name)
    if not target:
        return {"status": "failed", "error": f"Tool not found: {tool_name}"}

    try:
        if inspect.iscoroutinefunction(target):
            raw_result = await target(arguments)
        else:
            try:
                raw_result = target(**arguments)
            except TypeError:
                raw_result = target(arguments)
        result_dict = _coerce_tool_result_to_dict(raw_result)
        if "status" not in result_dict:
            result_dict["status"] = "ok"
        return result_dict
    except Exception as e:
        return {"status": "failed", "error": str(e)}


async def workflow_execute(kwargs_dict):
    kwargs = kwargs_dict or {}
    steps = kwargs.get("steps", [])
    stop_on_error = bool(kwargs.get("stop_on_error", True))
    max_steps = int(kwargs.get("max_steps", 30))
    max_steps = max(1, min(max_steps, 100))

    if not isinstance(steps, list) or not steps:
        return json.dumps({"status": "failed", "error": "steps must be a non-empty array"}, ensure_ascii=True)

    execution = []
    workflow_started = time.time()
    overall_status = "ok"

    for idx, step in enumerate(steps[:max_steps], start=1):
        if not isinstance(step, dict):
            execution.append({"step": idx, "status": "failed", "error": "Step must be object"})
            overall_status = "failed"
            if stop_on_error:
                break
            continue

        tool_name = str(step.get("tool_name", "")).strip()
        arguments = step.get("arguments", {}) or {}
        required = bool(step.get("required", True))

        if not isinstance(arguments, dict):
            result = {"status": "failed", "error": "arguments must be an object"}
        elif not tool_name:
            result = {"status": "failed", "error": "tool_name is required"}
        elif tool_name in {"workflow_execute"}:
            result = {"status": "failed", "error": "Recursive workflow_execute is blocked"}
        else:
            step_start = time.time()
            result = await _invoke_tool_by_name(tool_name, arguments)
            result["duration_ms"] = int((time.time() - step_start) * 1000)

        status = result.get("status", "ok")
        entry = {
            "step": idx,
            "tool_name": tool_name,
            "required": required,
            "status": status,
            "result": result,
        }
        execution.append(entry)

        failed = status not in {"ok", "success"}
        if failed and required:
            overall_status = "failed"
            if stop_on_error:
                break

    return json.dumps(
        {
            "status": overall_status,
            "executed_steps": len(execution),
            "duration_ms": int((time.time() - workflow_started) * 1000),
            "steps": execution,
        },
        ensure_ascii=True,
    )


async def task_autopilot(kwargs_dict):
    kwargs = kwargs_dict or {}
    objective = str(kwargs.get("objective", "")).strip()
    path_value = kwargs.get("path", ".")
    max_focus_files = int(kwargs.get("max_focus_files", 6))
    include_preview = bool(kwargs.get("include_preview", False))
    max_focus_files = max(1, min(max_focus_files, 20))

    if not objective:
        return json.dumps({"status": "failed", "error": "objective is required"}, ensure_ascii=True)

    plan = _coerce_tool_result_to_dict(await reasoning_plan({"goal": objective, "context": f"path={path_value}"}))
    base = _coerce_tool_result_to_dict(await codebase_analyze({"path": path_value, "max_files": 2000, "top_n_large_files": 20}))
    if base.get("status") != "ok":
        return json.dumps(
            {
                "status": "failed",
                "objective": objective,
                "plan": plan,
                "analysis_error": base,
            },
            ensure_ascii=True,
        )

    focus_files = []
    key_files = base.get("key_files", [])
    large_files = [item.get("path") for item in base.get("largest_files", []) if isinstance(item, dict)]
    for candidate in key_files + large_files:
        if not candidate:
            continue
        candidate_path = Path(candidate)
        if not _is_probably_text_source(candidate_path):
            continue
        if candidate not in focus_files:
            focus_files.append(candidate)
        if len(focus_files) >= max_focus_files:
            break

    deep_file_analysis = []
    for file_path in focus_files:
        analyzed = _coerce_tool_result_to_dict(
            await fs_analyze_file(
                {
                    "path": file_path,
                    "max_chars": 250000,
                    "include_preview": include_preview,
                }
            )
        )
        deep_file_analysis.append({"path": file_path, "analysis": analyzed})

    return json.dumps(
        {
            "status": "ok",
            "objective": objective,
            "path": path_value,
            "plan": plan,
            "codebase": base,
            "focus_files": focus_files,
            "deep_file_analysis": deep_file_analysis,
            "next_actions": [
                "Refine target files and run fs_patch/fs_edit_lines for code modifications.",
                "Use workflow_execute to run deterministic multi-step tool sequences.",
                "Run compile/tests with run_command after modifications.",
            ],
        },
        ensure_ascii=True,
    )


async def run_command(kwargs_dict):
    kwargs = kwargs_dict or {}
    command = str(kwargs.get("command", "")).strip()
    cwd_value = kwargs.get("cwd", ".")
    timeout_sec = float(kwargs.get("timeout_sec", 30))
    allow_dangerous = bool(kwargs.get("allow_dangerous", False))

    if not command:
        return json.dumps({"status": "failed", "error": "command is required"}, ensure_ascii=True)

    blocked_patterns = [
        r"\brm\s+-rf\b",
        r"\bdel\s+/f\b",
        r"\bformat\b",
        r"\bshutdown\b",
        r"\breboot\b",
        r"git\s+reset\s+--hard",
        r"remove-item\s+.+-recurse.+-force",
    ]
    lowered = command.lower()
    if not allow_dangerous:
        for pattern in blocked_patterns:
            if re.search(pattern, lowered):
                return json.dumps(
                    {
                        "status": "blocked",
                        "error": "Command blocked by safety policy. Set allow_dangerous=true only when explicitly intended.",
                        "command": command,
                    },
                    ensure_ascii=True,
                )

    try:
        cwd_path = _resolve_workspace_path(cwd_value, must_exist=True)
        if not cwd_path.is_dir():
            return json.dumps({"status": "failed", "error": f"cwd is not a directory: {cwd_path}"}, ensure_ascii=True)

        proc = subprocess.run(
            ["powershell", "-NoProfile", "-Command", command],
            cwd=str(cwd_path),
            capture_output=True,
            text=True,
            timeout=max(1.0, min(timeout_sec, 600.0)),
        )
        return json.dumps(
            {
                "status": "ok",
                "command": command,
                "cwd": cwd_path.relative_to(WORKSPACE_ROOT).as_posix() if cwd_path != WORKSPACE_ROOT else ".",
                "exit_code": proc.returncode,
                "stdout": proc.stdout[:20000],
                "stderr": proc.stderr[:20000],
            },
            ensure_ascii=True,
        )
    except subprocess.TimeoutExpired as e:
        return json.dumps(
            {
                "status": "failed",
                "error": f"Command timed out after {timeout_sec} seconds",
                "stdout": (e.stdout or "")[:8000],
                "stderr": (e.stderr or "")[:8000],
            },
            ensure_ascii=True,
        )
    except Exception as e:
        return json.dumps({"status": "failed", "error": str(e)}, ensure_ascii=True)


async def web_fetch(kwargs_dict):
    kwargs = kwargs_dict or {}
    url = str(kwargs.get("url", "")).strip()
    max_chars = int(kwargs.get("max_chars", 50000))
    extract_text = bool(kwargs.get("extract_text", True))
    timeout_sec = float(kwargs.get("timeout_sec", 20))
    max_chars = max(500, min(max_chars, 500000))

    if not url:
        return json.dumps({"status": "failed", "error": "url is required"}, ensure_ascii=True)
    if not re.match(r"^https?://", url, flags=re.IGNORECASE):
        return json.dumps({"status": "failed", "error": "Only http/https URLs are supported."}, ensure_ascii=True)

    try:
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "AgentLiveWeb/1.0 (+local)",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            },
        )
        with urllib.request.urlopen(req, timeout=max(1.0, min(timeout_sec, 120.0))) as response:
            body_bytes = response.read(max_chars * 4)
            charset = response.headers.get_content_charset() or "utf-8"
            body = body_bytes.decode(charset, errors="replace")
            body = body[:max_chars]
            text = _strip_html_to_text(body)[:max_chars] if extract_text else ""
            return json.dumps(
                {
                    "status": "ok",
                    "url": response.geturl(),
                    "status_code": response.status,
                    "content_type": response.headers.get("Content-Type", ""),
                    "title": _extract_title(body),
                    "body": body,
                    "text": text,
                },
                ensure_ascii=True,
            )
    except urllib.error.HTTPError as e:
        return json.dumps(
            {
                "status": "failed",
                "error": f"HTTPError: {e.code}",
                "url": url,
                "reason": str(e),
            },
            ensure_ascii=True,
        )
    except urllib.error.URLError as e:
        return json.dumps(
            {
                "status": "failed",
                "error": f"URLError: {e.reason}",
                "url": url,
            },
            ensure_ascii=True,
        )
    except Exception as e:
        return json.dumps({"status": "failed", "error": str(e), "url": url}, ensure_ascii=True)


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

    target = AVAILABLE_FUNCTIONS.get(tool_name)
    if not target:
        return json.dumps({"status": "failed", "error": f"Tool not found: {tool_name}"}, ensure_ascii=True)

    try:
        if inspect.iscoroutinefunction(target):
            result = await target(arguments)
        else:
            try:
                result = target(**arguments)
            except TypeError:
                result = target(arguments)
        return json.dumps({"status": "ok", "tool_name": tool_name, "result": str(result)}, ensure_ascii=True)
    except Exception as e:
        return json.dumps({"status": "failed", "tool_name": tool_name, "error": str(e)}, ensure_ascii=True)


AVAILABLE_FUNCTIONS.update(
    {
        "browser_tabs_list": browser_tabs_list,
        "browser_tab_select": browser_tab_select,
        "browser_close_blank_tabs": browser_close_blank_tabs,
        "fs_list": fs_list,
        "fs_read": fs_read,
        "fs_read_batch": fs_read_batch,
        "fs_edit_lines": fs_edit_lines,
        "fs_insert_lines": fs_insert_lines,
        "fs_write": fs_write,
        "fs_copy": fs_copy,
        "fs_move": fs_move,
        "fs_delete": fs_delete,
        "fs_patch": fs_patch,
        "fs_search": fs_search,
        "fs_analyze_file": fs_analyze_file,
        "codebase_analyze": codebase_analyze,
        "reasoning_plan": reasoning_plan,
        "memory_log": memory_log,
        "memory_search": memory_search,
        "memory_get": memory_get,
        "memory_promote": memory_promote,
        "memory_bootstrap": memory_bootstrap,
        "memory_reindex": memory_reindex,
        "tool_catalog": tool_catalog,
        "workflow_execute": workflow_execute,
        "task_autopilot": task_autopilot,
        "run_command": run_command,
        "web_fetch": web_fetch,
        "call_tool": call_tool,
    }
)


async def init_mcp_client():
    """Initializes the connection to the MCP Playwright server and loads its tools."""
    global mcp_session, _mcp_exit_stack
    if mcp_session is not None:
        return

    # Configure the MCP server command to use the local Playwright Edge script
    project_root = Path(__file__).resolve().parents[2]
    owner_file = project_root / ".playwright-mcp" / "active-owner.txt"
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
            "PLAYWRIGHT_MCP_USER_DATA_DIR": r"C:\Users\banot\AppData\Local\PlaywrightMCP\edge-profile",
            "PLAYWRIGHT_MCP_OUTPUT_DIR": r"C:\Users\banot\AppData\Local\PlaywrightMCP\output",
        }
    )

    mcp_launcher = project_root / "playwright-edge-mcp.js"
    if not mcp_launcher.exists():
        raise FileNotFoundError(f"MCP launcher not found: {mcp_launcher}")

    server_params = StdioServerParameters(command="node", args=[str(mcp_launcher)], env=mcp_env)

    print("Connecting to MCP Playwright server...")

    try:
        # Create long-lived managed transport/session contexts
        _mcp_exit_stack = AsyncExitStack()
        read, write = await _mcp_exit_stack.enter_async_context(stdio_client(server_params))
        mcp_session = await _mcp_exit_stack.enter_async_context(ClientSession(read, write))
        await mcp_session.initialize()

        # Discover available tools from the MCP server
        mcp_tools = await mcp_session.list_tools()

        print(f"Connected to MCP server! Found {len(mcp_tools.tools)} tools.")

        # Register the dynamically discovered tools with OpenAI's schema format
        for tool in mcp_tools.tools:
            properties = {}
            required = []
            if getattr(tool, "inputSchema", None) and "properties" in tool.inputSchema:
                properties = tool.inputSchema["properties"]
                required = tool.inputSchema.get("required", [])

            AGENT_TOOLS.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description or f"Tool provided by MCP: {tool.name}",
                        "parameters": {"type": "object", "properties": properties, "required": required},
                    },
                }
            )

            # Wrap each MCP tool with a strict execute -> verify -> recover loop.
            async def execute_mcp_tool(kwargs_dict, current_tool=tool.name):
                args = kwargs_dict or {}
                print(f"Executing MCP Tool --> {current_tool} with arguments: {args}")

                if current_tool.startswith("browser_") and current_tool not in OWNERSHIP_SKIP_TOOLS:
                    ownership = await _ensure_owned_working_tab(preserve_current_blank=(current_tool == "browser_navigate"))
                    if not ownership.get("ok", False):
                        return json.dumps(
                            {
                                "status": "failed",
                                "tool": current_tool,
                                "error": f"Tab ownership check failed: {ownership.get('error', 'unknown error')}",
                            },
                            ensure_ascii=True,
                        )

                before = await _capture_page_state(include_snapshot=current_tool in STATE_CHANGE_TOOLS)
                first = await _call_mcp_tool_raw(current_tool, args)
                first_verification = await _verify_step(current_tool, args, before, first)
                if first["ok"] and first_verification["ok"]:
                    return _format_step_response(current_tool, args, 1, first_verification, first, recovered=False)

                if current_tool not in RETRYABLE_TOOLS:
                    return _format_step_response(current_tool, args, 1, first_verification, first, recovered=False)

                await _call_mcp_tool_raw("browser_wait_for", {"time": 2})
                second_before = await _capture_page_state(include_snapshot=current_tool in STATE_CHANGE_TOOLS)
                second = await _call_mcp_tool_raw(current_tool, args)
                second_verification = await _verify_step(current_tool, args, second_before, second)
                return _format_step_response(
                    current_tool,
                    args,
                    2,
                    second_verification,
                    second,
                    recovered=second["ok"] and second_verification["ok"],
                )

            AVAILABLE_FUNCTIONS[tool.name] = execute_mcp_tool

        # Ensure we start from a clean owned working tab context.
        await _ensure_owned_working_tab()
        print("MCP tools successfully loaded into AGENT_TOOLS.")

    except Exception as e:
        print(f"Failed to connect to MCP server: {e}")
        print("Agent will proceed without external MCP tools.")
        if _mcp_exit_stack is not None:
            try:
                await _mcp_exit_stack.aclose()
            except Exception:
                pass
        _mcp_exit_stack = None
        mcp_session = None


async def shutdown_mcp_client():
    """Cleanly closes the MCP session/transport."""
    global mcp_session, _mcp_exit_stack
    if _mcp_exit_stack is not None:
        try:
            await _mcp_exit_stack.aclose()
        except Exception:
            # Agent should still exit even if MCP close emits transport warnings.
            pass
    _mcp_exit_stack = None
    mcp_session = None
