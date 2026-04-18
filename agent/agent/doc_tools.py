import json
import re
from pathlib import Path
try:
    from runtime_utils import redact_sensitive_data as _redact_sensitive_data, redact_sensitive_text as _redact_sensitive_text
except Exception:
    from .runtime_utils import redact_sensitive_data as _redact_sensitive_data, redact_sensitive_text as _redact_sensitive_text


class DocManager:
    """
    Documentation generation tools: docstrings, README sections, API docs,
    changelog entries, and inline comment generation.
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

    def _extract_functions_with_bodies(self, content: str, language: str) -> list:
        """Extract function signatures and their bodies for documentation."""
        functions = []
        if language == "python":
            for match in re.finditer(
                r"^((?:    )?(?:async\s+)?def\s+(\w+)\s*\(([^)]*)\)(?:\s*->\s*(\S+))?\s*:)\s*\n((?:(?:    )+.+\n)*)",
                content, re.MULTILINE
            ):
                sig = match.group(1).strip()
                name = match.group(2)
                params = match.group(3).strip()
                return_type = match.group(4) or ""
                body = match.group(5).strip()
                has_docstring = body.startswith('"""') or body.startswith("'''")
                line = content[:match.start()].count("\n") + 1
                functions.append({
                    "name": name, "signature": sig, "params": params,
                    "return_type": return_type, "has_docstring": has_docstring,
                    "body_preview": body[:200], "line": line,
                })
        elif language in {"javascript", "typescript"}:
            for match in re.finditer(
                r"(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*\(([^)]*)\)(?:\s*:\s*(\w+))?\s*\{",
                content, re.MULTILINE
            ):
                name = match.group(1)
                params = match.group(2).strip()
                return_type = match.group(3) or ""
                # Check for JSDoc above
                before = content[:match.start()]
                has_jsdoc = bool(re.search(r"/\*\*[\s\S]*?\*/\s*$", before))
                line = before.count("\n") + 1
                functions.append({
                    "name": name, "params": params, "return_type": return_type,
                    "has_docstring": has_jsdoc, "line": line,
                })
        return functions

    async def generate_docstrings(self, kwargs_dict):
        """Analyze a file and generate missing docstrings/JSDoc comments."""
        kwargs = kwargs_dict or {}
        path_value = str(kwargs.get("path", "")).strip()
        dry_run = bool(kwargs.get("dry_run", True))

        if not path_value:
            return self._json_response({"status": "failed", "error": "path is required"})

        try:
            file_path = self.resolve_workspace_path(path_value, must_exist=True)
        except Exception as e:
            return self._json_response({"status": "failed", "error": str(e)})

        content = file_path.read_text(encoding="utf-8", errors="replace")
        language = self._detect_language(file_path)
        functions = self._extract_functions_with_bodies(content, language)

        missing = [f for f in functions if not f.get("has_docstring")]
        documented = [f for f in functions if f.get("has_docstring")]

        # Generate docstring templates for missing ones
        generated = []
        for fn in missing:
            if language == "python":
                params_list = [p.strip().split(":")[0].split("=")[0].strip()
                               for p in fn.get("params", "").split(",")
                               if p.strip() and p.strip() != "self" and p.strip() != "cls"]
                param_docs = "\n".join(f"        {p}: Description." for p in params_list)
                returns = f"\n\n    Returns:\n        {fn.get('return_type') or 'Result'}." if fn.get("return_type") else ""
                docstring = f'    """{fn["name"]}: TODO describe purpose.\n\n    Args:\n{param_docs}{returns}\n    """'
                generated.append({"name": fn["name"], "line": fn["line"], "docstring": docstring})
            elif language in {"javascript", "typescript"}:
                params_list = [p.strip().split("=")[0].split(":")[0].strip()
                               for p in fn.get("params", "").split(",") if p.strip()]
                param_docs = "\n".join(f" * @param {{any}} {p} - Description." for p in params_list)
                returns = f"\n * @returns {{{fn.get('return_type', 'any')}}} Result." if fn.get("return_type") else ""
                jsdoc = f"/**\n * {fn['name']}: TODO describe purpose.\n{param_docs}{returns}\n */"
                generated.append({"name": fn["name"], "line": fn["line"], "docstring": jsdoc})

        return self._json_response({
            "status": "ok",
            "path": file_path.relative_to(self.workspace_root).as_posix(),
            "language": language,
            "total_functions": len(functions),
            "documented": len(documented),
            "missing_docs": len(missing),
            "coverage_percent": round(len(documented) / max(1, len(functions)) * 100, 1),
            "generated_docstrings": generated,
            "dry_run": dry_run,
        })

    async def generate_changelog_entry(self, kwargs_dict=None):
        """Generate a changelog entry from recent git commits."""
        kwargs = kwargs_dict or {}
        count = max(1, min(int(kwargs.get("count", 20)), 100))
        version = str(kwargs.get("version", "")).strip()
        categories = {"feat": [], "fix": [], "refactor": [], "docs": [], "test": [], "chore": [], "other": []}

        import subprocess
        try:
            result = subprocess.run(
                ["git", "log", f"-{count}", "--oneline"],
                cwd=str(self.workspace_root),
                capture_output=True, text=True, timeout=15,
            )
            if result.returncode != 0:
                return self._json_response({"status": "failed", "error": result.stderr})
        except Exception as e:
            return self._json_response({"status": "failed", "error": str(e)})

        for line in result.stdout.strip().splitlines():
            parts = line.split(" ", 1)
            if len(parts) < 2:
                continue
            commit_hash = parts[0]
            message = parts[1]

            # Categorize by conventional commit prefix
            categorized = False
            for prefix in ["feat", "fix", "refactor", "docs", "test", "chore"]:
                if message.lower().startswith(f"{prefix}:") or message.lower().startswith(f"{prefix}("):
                    categories[prefix].append(message)
                    categorized = True
                    break
            if not categorized:
                categories["other"].append(message)

        # Build changelog markdown
        lines = []
        if version:
            lines.append(f"## {version}")
        else:
            from datetime import datetime
            lines.append(f"## [{datetime.now().strftime('%Y-%m-%d')}]")
        lines.append("")

        section_titles = {
            "feat": "Features", "fix": "Bug Fixes", "refactor": "Refactoring",
            "docs": "Documentation", "test": "Tests", "chore": "Maintenance", "other": "Other",
        }
        for key, title in section_titles.items():
            if categories[key]:
                lines.append(f"### {title}")
                for msg in categories[key]:
                    lines.append(f"- {msg}")
                lines.append("")

        changelog_text = "\n".join(lines).strip()

        return self._json_response({
            "status": "ok",
            "version": version,
            "total_commits": sum(len(v) for v in categories.values()),
            "categories": {k: len(v) for k, v in categories.items()},
            "changelog": changelog_text,
        })

    async def doc_coverage(self, kwargs_dict=None):
        """Analyze documentation coverage across the project."""
        kwargs = kwargs_dict or {}
        path_value = str(kwargs.get("path", ".")).strip()

        try:
            root = self.resolve_workspace_path(path_value, must_exist=True)
        except Exception as e:
            return self._json_response({"status": "failed", "error": str(e)})

        total_functions = 0
        documented_functions = 0
        files_report = []

        for file_path in root.rglob("*"):
            if not file_path.is_file():
                continue
            language = self._detect_language(file_path)
            if language == "unknown":
                continue
            if any(part in {"node_modules", "__pycache__", ".venv", ".git"} for part in file_path.parts):
                continue

            try:
                content = file_path.read_text(encoding="utf-8", errors="replace")
                functions = self._extract_functions_with_bodies(content, language)
                if not functions:
                    continue
                doc_count = sum(1 for f in functions if f.get("has_docstring"))
                total_functions += len(functions)
                documented_functions += doc_count
                rel = file_path.relative_to(self.workspace_root).as_posix()
                if doc_count < len(functions):
                    undoc = [f["name"] for f in functions if not f.get("has_docstring")]
                    files_report.append({
                        "path": rel,
                        "functions": len(functions),
                        "documented": doc_count,
                        "undocumented_names": undoc[:20],
                    })
            except Exception:
                continue

        files_report.sort(key=lambda x: x["functions"] - x["documented"], reverse=True)

        return self._json_response({
            "status": "ok",
            "total_functions": total_functions,
            "documented": documented_functions,
            "undocumented": total_functions - documented_functions,
            "coverage_percent": round(documented_functions / max(1, total_functions) * 100, 1),
            "files_needing_docs": files_report[:30],
        })
