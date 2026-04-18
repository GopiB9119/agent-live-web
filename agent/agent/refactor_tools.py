import json
import re
import hashlib
from pathlib import Path
try:
    from runtime_utils import redact_sensitive_data as _redact_sensitive_data, redact_sensitive_text as _redact_sensitive_text
except Exception:
    from .runtime_utils import redact_sensitive_data as _redact_sensitive_data, redact_sensitive_text as _redact_sensitive_text


class RefactorManager:
    """
    Code refactoring tools: rename symbol, extract function, find dead code,
    find duplicates, and suggest improvements.
    """

    def __init__(self, workspace_root: Path, resolve_workspace_path_fn, fs_manager=None):
        self.workspace_root = Path(workspace_root).resolve()
        self.resolve_workspace_path = resolve_workspace_path_fn
        self.fs_manager = fs_manager

    @staticmethod
    def _json_response(payload, max_chars=30000):
        return json.dumps(_redact_sensitive_data(payload, max_chars=max_chars), ensure_ascii=True)

    def _detect_language(self, path_obj: Path) -> str:
        return {
            ".py": "python", ".js": "javascript", ".ts": "typescript",
            ".jsx": "javascript", ".tsx": "typescript",
        }.get(path_obj.suffix.lower(), "unknown")

    async def rename_symbol(self, kwargs_dict):
        """Rename a symbol (function/variable/class) across files."""
        kwargs = kwargs_dict or {}
        old_name = str(kwargs.get("old_name", "")).strip()
        new_name = str(kwargs.get("new_name", "")).strip()
        path_value = str(kwargs.get("path", ".")).strip()
        file_glob = str(kwargs.get("file_glob", "*")).strip()
        dry_run = bool(kwargs.get("dry_run", True))
        whole_word = bool(kwargs.get("whole_word", True))

        if not old_name or not new_name:
            return self._json_response({"status": "failed", "error": "old_name and new_name are required"})
        if not re.match(r"^[A-Za-z_]\w*$", new_name):
            return self._json_response({"status": "failed", "error": f"Invalid identifier: {new_name}"})

        try:
            root = self.resolve_workspace_path(path_value, must_exist=True)
        except Exception as e:
            return self._json_response({"status": "failed", "error": str(e)})

        pattern = re.compile(rf"\b{re.escape(old_name)}\b" if whole_word else re.escape(old_name))
        affected_files = []
        total_replacements = 0

        files = root.rglob("*") if root.is_dir() else [root]
        for file_path in files:
            if not file_path.is_file():
                continue
            if file_path.suffix.lower() not in {".py", ".js", ".ts", ".jsx", ".tsx", ".md", ".json"}:
                continue
            if any(part.startswith(".") or part in {"node_modules", "__pycache__", ".venv"} for part in file_path.parts):
                continue

            try:
                content = file_path.read_text(encoding="utf-8", errors="replace")
                matches = pattern.findall(content)
                if not matches:
                    continue
                count = len(matches)
                new_content = pattern.sub(new_name, content)
                rel = file_path.relative_to(self.workspace_root).as_posix()
                affected_files.append({"path": rel, "replacements": count})
                total_replacements += count

                if not dry_run:
                    file_path.write_text(new_content, encoding="utf-8")
            except Exception:
                continue

        return self._json_response({
            "status": "ok",
            "old_name": old_name,
            "new_name": new_name,
            "dry_run": dry_run,
            "files_affected": len(affected_files),
            "total_replacements": total_replacements,
            "affected_files": affected_files,
        })

    async def find_dead_code(self, kwargs_dict=None):
        """Find functions/variables defined but never referenced elsewhere."""
        kwargs = kwargs_dict or {}
        path_value = str(kwargs.get("path", ".")).strip()
        language = str(kwargs.get("language", "")).strip()

        try:
            root = self.resolve_workspace_path(path_value, must_exist=True)
        except Exception as e:
            return self._json_response({"status": "failed", "error": str(e)})

        # Collect all definitions
        definitions = {}
        all_content = ""
        files = root.rglob("*.py") if not language or language == "python" else root.rglob("*.js")

        for file_path in files:
            if any(part in {"node_modules", "__pycache__", ".venv", "tests", ".git"} for part in file_path.parts):
                continue
            try:
                content = file_path.read_text(encoding="utf-8", errors="replace")
                all_content += content + "\n"
                rel = file_path.relative_to(self.workspace_root).as_posix()

                if file_path.suffix == ".py":
                    for match in re.finditer(r"^(?:    )?def\s+(\w+)\s*\(", content, re.MULTILINE):
                        name = match.group(1)
                        if not name.startswith("_"):
                            definitions[name] = {"file": rel, "line": content[:match.start()].count("\n") + 1}
                elif file_path.suffix in {".js", ".ts"}:
                    for match in re.finditer(r"(?:function|const|let|var)\s+(\w+)", content, re.MULTILINE):
                        name = match.group(1)
                        definitions[name] = {"file": rel, "line": content[:match.start()].count("\n") + 1}
            except Exception:
                continue

        # Find unreferenced definitions
        dead = []
        for name, info in definitions.items():
            # Count occurrences (subtract 1 for the definition itself)
            count = len(re.findall(rf"\b{re.escape(name)}\b", all_content))
            if count <= 1:
                dead.append({"name": name, **info, "references": 0})

        dead.sort(key=lambda x: x["file"])

        return self._json_response({
            "status": "ok",
            "total_definitions": len(definitions),
            "dead_code_count": len(dead),
            "dead_code": dead[:100],
        })

    async def find_duplicates(self, kwargs_dict=None):
        """Find duplicate or near-duplicate code blocks."""
        kwargs = kwargs_dict or {}
        path_value = str(kwargs.get("path", ".")).strip()
        min_lines = max(3, int(kwargs.get("min_lines", 5)))

        try:
            root = self.resolve_workspace_path(path_value, must_exist=True)
        except Exception as e:
            return self._json_response({"status": "failed", "error": str(e)})

        # Collect blocks from source files
        blocks = {}
        for file_path in root.rglob("*"):
            if not file_path.is_file():
                continue
            if file_path.suffix.lower() not in {".py", ".js", ".ts"}:
                continue
            if any(part in {"node_modules", "__pycache__", ".venv", ".git"} for part in file_path.parts):
                continue

            try:
                content = file_path.read_text(encoding="utf-8", errors="replace")
                lines = content.splitlines()
                rel = file_path.relative_to(self.workspace_root).as_posix()

                for i in range(len(lines) - min_lines + 1):
                    block = "\n".join(line.strip() for line in lines[i:i + min_lines] if line.strip())
                    if len(block) < 20:
                        continue
                    block_hash = hashlib.sha1(block.encode()).hexdigest()[:12]
                    if block_hash not in blocks:
                        blocks[block_hash] = []
                    blocks[block_hash].append({"file": rel, "line": i + 1})
            except Exception:
                continue

        duplicates = []
        for block_hash, locations in blocks.items():
            if len(locations) > 1:
                # Deduplicate nearby locations in same file
                unique_locs = []
                for loc in locations:
                    if not unique_locs or unique_locs[-1]["file"] != loc["file"] or abs(unique_locs[-1]["line"] - loc["line"]) > min_lines:
                        unique_locs.append(loc)
                if len(unique_locs) > 1:
                    duplicates.append({"hash": block_hash, "count": len(unique_locs), "locations": unique_locs[:10]})

        duplicates.sort(key=lambda x: x["count"], reverse=True)

        return self._json_response({
            "status": "ok",
            "duplicate_groups": len(duplicates),
            "duplicates": duplicates[:50],
        })

    async def code_metrics(self, kwargs_dict=None):
        """Compute code complexity and quality metrics for a file."""
        kwargs = kwargs_dict or {}
        path_value = str(kwargs.get("path", "")).strip()

        if not path_value:
            return self._json_response({"status": "failed", "error": "path is required"})

        try:
            file_path = self.resolve_workspace_path(path_value, must_exist=True)
        except Exception as e:
            return self._json_response({"status": "failed", "error": str(e)})

        content = file_path.read_text(encoding="utf-8", errors="replace")
        lines = content.splitlines()
        language = self._detect_language(file_path)

        total_lines = len(lines)
        blank_lines = sum(1 for line in lines if not line.strip())
        comment_lines = 0
        for line in lines:
            stripped = line.strip()
            if language == "python" and stripped.startswith("#"):
                comment_lines += 1
            elif language in {"javascript", "typescript"} and (stripped.startswith("//") or stripped.startswith("/*")):
                comment_lines += 1

        code_lines = total_lines - blank_lines - comment_lines

        # Count functions and classes
        if language == "python":
            functions = len(re.findall(r"^\s*(?:async\s+)?def\s+\w+", content, re.MULTILINE))
            classes = len(re.findall(r"^\s*class\s+\w+", content, re.MULTILINE))
            max_indent = max((len(line) - len(line.lstrip()) for line in lines if line.strip()), default=0)
            nesting_depth = max_indent // 4
        else:
            functions = len(re.findall(r"(?:function|async\s+function|=>\s*\{)", content))
            classes = len(re.findall(r"\bclass\s+\w+", content))
            nesting_depth = max(line.count("{") for line in lines) if lines else 0

        # Simple cyclomatic complexity estimate (branches)
        branches = len(re.findall(r"\b(if|elif|else|for|while|catch|except|case|switch|&&|\|\|)\b", content))
        complexity = "low" if branches < 10 else "medium" if branches < 25 else "high"

        # Long function detection
        long_functions = []
        if language == "python":
            func_starts = [(m.start(), m.group()) for m in re.finditer(r"^(?:    )?(?:async\s+)?def\s+(\w+)", content, re.MULTILINE)]
            for i, (start, sig) in enumerate(func_starts):
                end = func_starts[i + 1][0] if i + 1 < len(func_starts) else len(content)
                func_lines = content[start:end].count("\n") + 1
                name = re.search(r"def\s+(\w+)", sig).group(1)
                if func_lines > 50:
                    long_functions.append({"name": name, "lines": func_lines})

        return self._json_response({
            "status": "ok",
            "path": file_path.relative_to(self.workspace_root).as_posix(),
            "language": language,
            "total_lines": total_lines,
            "code_lines": code_lines,
            "blank_lines": blank_lines,
            "comment_lines": comment_lines,
            "functions": functions,
            "classes": classes,
            "branches": branches,
            "complexity": complexity,
            "max_nesting_depth": nesting_depth,
            "long_functions": long_functions[:20],
        })
