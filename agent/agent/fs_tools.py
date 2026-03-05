import fnmatch
import hashlib
import json
import re
import shutil
from pathlib import Path


class FSManager:
    """
    Filesystem and codebase-analysis manager for workspace-scoped operations.
    """

    def __init__(
        self,
        workspace_root: Path,
        resolve_workspace_path_fn,
        noise_dir_names=None,
        binary_suffixes=None,
    ):
        self.workspace_root = Path(workspace_root).resolve()
        self.resolve_workspace_path = resolve_workspace_path_fn
        self.noise_dir_names = set(noise_dir_names or [])
        self.binary_suffixes = set(binary_suffixes or [])

    @staticmethod
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

    @staticmethod
    def _extract_python_symbols(content: str):
        imports = []
        for line in content.splitlines():
            stripped = line.strip()
            if stripped.startswith("import "):
                imports.append(stripped)
            elif stripped.startswith("from ") and " import " in stripped:
                imports.append(stripped)

        functions = re.findall(r"^\s*def\s+([A-Za-z_]\w*)\s*\(", content, flags=re.MULTILINE)
        classes = re.findall(r"^\s*class\s+([A-Za-z_]\w*)\b", content, flags=re.MULTILINE)
        return {"imports": imports[:120], "functions": functions[:200], "classes": classes[:200]}

    @staticmethod
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

    def _analyze_content(self, path_obj: Path, content: str):
        language = self._language_from_path(path_obj)
        lines = content.splitlines()
        line_count = len(lines)

        if language == "python":
            symbol_data = self._extract_python_symbols(content)
        elif language in {"javascript", "typescript"}:
            symbol_data = self._extract_js_ts_symbols(content)
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

    def _is_noise_path(self, path_obj: Path) -> bool:
        return any(part in self.noise_dir_names for part in path_obj.parts)

    def is_probably_text_source(self, path_obj: Path) -> bool:
        if self._is_noise_path(path_obj):
            return False
        if path_obj.suffix.lower() in self.binary_suffixes:
            return False
        return True

    async def fs_list(self, kwargs_dict=None):
        kwargs = kwargs_dict or {}
        path_value = kwargs.get("path", ".")
        recursive = bool(kwargs.get("recursive", False))
        include_hidden = bool(kwargs.get("include_hidden", False))
        max_entries = int(kwargs.get("max_entries", 200))
        max_entries = max(1, min(max_entries, 2000))

        try:
            root = self.resolve_workspace_path(path_value, must_exist=True)
            if not root.is_dir():
                return json.dumps({"status": "failed", "error": f"Not a directory: {root}"}, ensure_ascii=True)

            iterator = root.rglob("*") if recursive else root.iterdir()
            entries = []
            for item in iterator:
                rel = item.relative_to(self.workspace_root).as_posix()
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
                    "root": root.relative_to(self.workspace_root).as_posix() if root != self.workspace_root else ".",
                    "count": len(entries),
                    "entries": entries,
                },
                ensure_ascii=True,
            )
        except Exception as e:
            return json.dumps({"status": "failed", "error": str(e)}, ensure_ascii=True)

    async def fs_read(self, kwargs_dict):
        kwargs = kwargs_dict or {}
        path_value = kwargs.get("path")
        encoding = kwargs.get("encoding", "utf-8")
        max_chars = int(kwargs.get("max_chars", 20000))
        max_chars = max(200, min(max_chars, 500000))

        try:
            file_path = self.resolve_workspace_path(path_value, must_exist=True)
            if not file_path.is_file():
                return json.dumps({"status": "failed", "error": f"Not a file: {file_path}"}, ensure_ascii=True)

            content = file_path.read_text(encoding=encoding, errors="replace")
            truncated = len(content) > max_chars
            payload = content[:max_chars]
            return json.dumps(
                {
                    "status": "ok",
                    "path": file_path.relative_to(self.workspace_root).as_posix(),
                    "chars": len(content),
                    "truncated": truncated,
                    "content": payload,
                },
                ensure_ascii=True,
            )
        except Exception as e:
            return json.dumps({"status": "failed", "error": str(e)}, ensure_ascii=True)

    async def fs_read_batch(self, kwargs_dict):
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
                file_path = self.resolve_workspace_path(raw_path, must_exist=True)
                if not file_path.is_file():
                    results.append({"path": str(raw_path), "status": "failed", "error": "Not a file"})
                    failures.append(str(raw_path))
                    continue
                content = file_path.read_text(encoding=encoding, errors="replace")
                rel_path = file_path.relative_to(self.workspace_root).as_posix()
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

    async def fs_edit_lines(self, kwargs_dict):
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

            file_path = self.resolve_workspace_path(path_value, must_exist=True)
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
                    "path": file_path.relative_to(self.workspace_root).as_posix(),
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

    async def fs_insert_lines(self, kwargs_dict):
        kwargs = kwargs_dict or {}
        path_value = kwargs.get("path")
        line = int(kwargs.get("line", 0))
        content = str(kwargs.get("content", ""))
        encoding = kwargs.get("encoding", "utf-8")
        dry_run = bool(kwargs.get("dry_run", False))

        try:
            if line < 1:
                return json.dumps({"status": "failed", "error": "line must be >= 1"}, ensure_ascii=True)

            file_path = self.resolve_workspace_path(path_value, must_exist=True)
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
                    "path": file_path.relative_to(self.workspace_root).as_posix(),
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

    async def fs_write(self, kwargs_dict):
        kwargs = kwargs_dict or {}
        path_value = kwargs.get("path")
        content = str(kwargs.get("content", ""))
        append = bool(kwargs.get("append", False))
        encoding = kwargs.get("encoding", "utf-8")

        try:
            file_path = self.resolve_workspace_path(path_value, must_exist=False)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            mode = "a" if append else "w"
            with file_path.open(mode, encoding=encoding, errors="replace", newline="") as fh:
                fh.write(content)
            return json.dumps(
                {
                    "status": "ok",
                    "path": file_path.relative_to(self.workspace_root).as_posix(),
                    "append": append,
                    "written_chars": len(content),
                },
                ensure_ascii=True,
            )
        except Exception as e:
            return json.dumps({"status": "failed", "error": str(e)}, ensure_ascii=True)

    async def fs_copy(self, kwargs_dict):
        kwargs = kwargs_dict or {}
        source_value = kwargs.get("source")
        destination_value = kwargs.get("destination")
        overwrite = bool(kwargs.get("overwrite", False))

        try:
            source_path = self.resolve_workspace_path(source_value, must_exist=True)
            destination_path = self.resolve_workspace_path(destination_value, must_exist=False)

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
                    "source": source_path.relative_to(self.workspace_root).as_posix(),
                    "destination": destination_path.relative_to(self.workspace_root).as_posix(),
                    "overwrite": overwrite,
                },
                ensure_ascii=True,
            )
        except Exception as e:
            return json.dumps({"status": "failed", "error": str(e)}, ensure_ascii=True)

    async def fs_move(self, kwargs_dict):
        kwargs = kwargs_dict or {}
        source_value = kwargs.get("source")
        destination_value = kwargs.get("destination")
        overwrite = bool(kwargs.get("overwrite", False))

        try:
            source_path = self.resolve_workspace_path(source_value, must_exist=True)
            destination_path = self.resolve_workspace_path(destination_value, must_exist=False)

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
                    "source": source_path.relative_to(self.workspace_root).as_posix(),
                    "destination": destination_path.relative_to(self.workspace_root).as_posix(),
                    "overwrite": overwrite,
                },
                ensure_ascii=True,
            )
        except Exception as e:
            return json.dumps({"status": "failed", "error": str(e)}, ensure_ascii=True)

    async def fs_delete(self, kwargs_dict):
        kwargs = kwargs_dict or {}
        path_value = kwargs.get("path")
        recursive = bool(kwargs.get("recursive", False))
        missing_ok = bool(kwargs.get("missing_ok", False))

        try:
            target_path = self.resolve_workspace_path(path_value, must_exist=False)

            if not target_path.exists():
                if missing_ok:
                    return json.dumps(
                        {
                            "status": "ok",
                            "path": target_path.relative_to(self.workspace_root).as_posix(),
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
                    "path": target_path.relative_to(self.workspace_root).as_posix(),
                    "deleted": True,
                    "type": deleted_type,
                    "recursive": recursive,
                },
                ensure_ascii=True,
            )
        except Exception as e:
            return json.dumps({"status": "failed", "error": str(e)}, ensure_ascii=True)

    async def fs_patch(self, kwargs_dict):
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

            file_path = self.resolve_workspace_path(path_value, must_exist=False)
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
                        "path": file_path.relative_to(self.workspace_root).as_posix(),
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
                    "path": file_path.relative_to(self.workspace_root).as_posix(),
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

    async def fs_search(self, kwargs_dict):
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

            root = self.resolve_workspace_path(path_value, must_exist=True)
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
                                "path": file_path.relative_to(self.workspace_root).as_posix(),
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
                    "root": root.relative_to(self.workspace_root).as_posix() if root != self.workspace_root else ".",
                    "count": len(matches),
                    "matches": matches,
                },
                ensure_ascii=True,
            )
        except Exception as e:
            return json.dumps({"status": "failed", "error": str(e)}, ensure_ascii=True)

    async def fs_analyze_file(self, kwargs_dict):
        kwargs = kwargs_dict or {}
        path_value = kwargs.get("path")
        encoding = kwargs.get("encoding", "utf-8")
        max_chars = int(kwargs.get("max_chars", 200000))
        include_preview = bool(kwargs.get("include_preview", True))
        max_chars = max(500, min(max_chars, 2000000))

        try:
            file_path = self.resolve_workspace_path(path_value, must_exist=True)
            if not file_path.is_file():
                return json.dumps({"status": "failed", "error": f"Not a file: {file_path}"}, ensure_ascii=True)

            content = file_path.read_text(encoding=encoding, errors="replace")
            truncated = len(content) > max_chars
            parse_content = content[:max_chars]
            analysis = self._analyze_content(file_path, parse_content)

            payload = {
                "status": "ok",
                "path": file_path.relative_to(self.workspace_root).as_posix(),
                "chars": len(content),
                "truncated_for_analysis": truncated,
                "analysis": analysis,
            }
            if include_preview:
                payload["preview"] = parse_content[:1200]
            return json.dumps(payload, ensure_ascii=True)
        except Exception as e:
            return json.dumps({"status": "failed", "error": str(e)}, ensure_ascii=True)

    async def codebase_analyze(self, kwargs_dict=None):
        kwargs = kwargs_dict or {}
        path_value = kwargs.get("path", ".")
        max_files = int(kwargs.get("max_files", 1200))
        include_hidden = bool(kwargs.get("include_hidden", False))
        top_n_large_files = int(kwargs.get("top_n_large_files", 20))
        max_files = max(10, min(max_files, 10000))
        top_n_large_files = max(3, min(top_n_large_files, 100))

        try:
            root = self.resolve_workspace_path(path_value, must_exist=True)
            if not root.is_dir():
                return json.dumps({"status": "failed", "error": f"Not a directory: {root}"}, ensure_ascii=True)

            files = []
            dirs_set = set()
            lang_counts = {}
            total_size = 0

            for item in root.rglob("*"):
                rel = item.relative_to(self.workspace_root).as_posix()
                if not include_hidden and any(part.startswith(".") for part in Path(rel).parts):
                    continue
                if self._is_noise_path(Path(rel)):
                    continue

                if item.is_dir():
                    dirs_set.add(rel)
                    continue
                if not item.is_file():
                    continue
                if not self.is_probably_text_source(Path(rel)):
                    continue

                try:
                    size = item.stat().st_size
                except Exception:
                    size = 0
                total_size += size
                language = self._language_from_path(item)
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
                    "root": root.relative_to(self.workspace_root).as_posix() if root != self.workspace_root else ".",
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
