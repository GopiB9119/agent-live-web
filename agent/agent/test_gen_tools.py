import json
import os
import re
import subprocess
from pathlib import Path
try:
    from runtime_utils import redact_sensitive_data as _redact_sensitive_data, redact_sensitive_text as _redact_sensitive_text
except Exception:
    from .runtime_utils import redact_sensitive_data as _redact_sensitive_data, redact_sensitive_text as _redact_sensitive_text


class TestGenManager:
    """
    Test generation, execution, and coverage manager.
    Generates test skeletons, runs tests, parses results, and suggests coverage gaps.
    """

    def __init__(self, workspace_root: Path, resolve_workspace_path_fn, fs_manager=None):
        self.workspace_root = Path(workspace_root).resolve()
        self.resolve_workspace_path = resolve_workspace_path_fn
        self.fs_manager = fs_manager

    @staticmethod
    def _json_response(payload, max_chars=30000):
        return json.dumps(_redact_sensitive_data(payload, max_chars=max_chars), ensure_ascii=True)

    def _detect_language(self, path_obj: Path) -> str:
        suffix = path_obj.suffix.lower()
        return {
            ".py": "python", ".js": "javascript", ".ts": "typescript",
            ".jsx": "javascript", ".tsx": "typescript",
        }.get(suffix, "unknown")

    def _detect_test_framework(self, language: str) -> dict:
        if language == "python":
            has_pytest = (self.workspace_root / "pytest.ini").exists() or (self.workspace_root / "pyproject.toml").exists()
            return {"framework": "pytest" if has_pytest else "unittest", "runner": "python -m pytest" if has_pytest else "python -m unittest"}
        elif language in {"javascript", "typescript"}:
            pkg_path = self.workspace_root / "package.json"
            if pkg_path.exists():
                try:
                    pkg = json.loads(pkg_path.read_text(encoding="utf-8"))
                    deps = {**pkg.get("dependencies", {}), **pkg.get("devDependencies", {})}
                    if "jest" in deps:
                        return {"framework": "jest", "runner": "npx jest"}
                    if "vitest" in deps:
                        return {"framework": "vitest", "runner": "npx vitest"}
                    if "mocha" in deps:
                        return {"framework": "mocha", "runner": "npx mocha"}
                except Exception:
                    pass
            return {"framework": "node:test", "runner": "node --test"}
        return {"framework": "unknown", "runner": ""}

    def _extract_functions_for_testing(self, content: str, language: str) -> list:
        """Extract function/method signatures that should be tested."""
        functions = []
        if language == "python":
            for match in re.finditer(r"^(?:    )?(?:async\s+)?def\s+(\w+)\s*\(([^)]*)\)", content, re.MULTILINE):
                name = match.group(1)
                if not name.startswith("_"):
                    params = [p.strip().split(":")[0].split("=")[0].strip() for p in match.group(2).split(",") if p.strip() and p.strip() != "self"]
                    functions.append({"name": name, "params": params, "line": content[:match.start()].count("\n") + 1})
            for match in re.finditer(r"^class\s+(\w+)", content, re.MULTILINE):
                functions.append({"name": match.group(1), "params": [], "type": "class", "line": content[:match.start()].count("\n") + 1})
        elif language in {"javascript", "typescript"}:
            for match in re.finditer(r"(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*\(([^)]*)\)", content, re.MULTILINE):
                name = match.group(1)
                params = [p.strip().split("=")[0].split(":")[0].strip() for p in match.group(2).split(",") if p.strip()]
                functions.append({"name": name, "params": params, "line": content[:match.start()].count("\n") + 1})
            for match in re.finditer(r"(?:export\s+)?(?:const|let)\s+(\w+)\s*=\s*(?:async\s*)?\(", content, re.MULTILINE):
                functions.append({"name": match.group(1), "params": [], "line": content[:match.start()].count("\n") + 1})
            for match in re.finditer(r"^class\s+(\w+)", content, re.MULTILINE):
                functions.append({"name": match.group(1), "params": [], "type": "class", "line": content[:match.start()].count("\n") + 1})
        return functions

    def _generate_python_test_skeleton(self, source_path: Path, functions: list, framework: str) -> str:
        rel = source_path.relative_to(self.workspace_root).as_posix()
        module = source_path.stem
        lines = []
        if framework == "pytest":
            lines.append(f'"""Auto-generated tests for {rel}"""')
            lines.append(f"import {module}\n\n")
            for fn in functions:
                if fn.get("type") == "class":
                    lines.append(f"class Test{fn['name']}:")
                    lines.append(f"    def test_{fn['name'].lower()}_creation(self):")
                    lines.append(f"        instance = {module}.{fn['name']}()")
                    lines.append(f"        assert instance is not None\n")
                else:
                    param_str = ", ".join(f"None" for _ in fn["params"])
                    lines.append(f"def test_{fn['name']}():")
                    lines.append(f"    result = {module}.{fn['name']}({param_str})")
                    lines.append(f"    assert result is not None\n")
        else:
            lines.append(f'"""Auto-generated tests for {rel}"""')
            lines.append("import unittest")
            lines.append(f"import {module}\n\n")
            class_name = f"Test{module.replace('_', ' ').title().replace(' ', '')}"
            lines.append(f"class {class_name}(unittest.TestCase):")
            for fn in functions:
                if fn.get("type") == "class":
                    lines.append(f"    def test_{fn['name'].lower()}_creation(self):")
                    lines.append(f"        instance = {module}.{fn['name']}()")
                    lines.append(f"        self.assertIsNotNone(instance)\n")
                else:
                    lines.append(f"    def test_{fn['name']}(self):")
                    param_str = ", ".join("None" for _ in fn["params"])
                    lines.append(f"        result = {module}.{fn['name']}({param_str})")
                    lines.append(f"        self.assertIsNotNone(result)\n")
        return "\n".join(lines) + "\n"

    def _generate_js_test_skeleton(self, source_path: Path, functions: list, framework: str) -> str:
        rel = source_path.relative_to(self.workspace_root).as_posix()
        module_path = f"./{source_path.stem}"
        lines = []
        if framework == "node:test":
            lines.append(f"// Auto-generated tests for {rel}")
            lines.append("const { describe, it } = require('node:test');")
            lines.append("const assert = require('node:assert/strict');")
            lines.append(f"const mod = require('{module_path}');\n")
            lines.append(f"describe('{source_path.stem}', () => {{")
            for fn in functions:
                if fn.get("type") == "class":
                    lines.append(f"  it('{fn['name']} can be instantiated', () => {{")
                    lines.append(f"    const instance = new mod.{fn['name']}();")
                    lines.append(f"    assert.ok(instance);")
                    lines.append(f"  }});\n")
                else:
                    lines.append(f"  it('{fn['name']} returns a value', () => {{")
                    lines.append(f"    const result = mod.{fn['name']}();")
                    lines.append(f"    assert.ok(result !== undefined);")
                    lines.append(f"  }});\n")
            lines.append("});")
        else:
            lines.append(f"// Auto-generated tests for {rel}")
            lines.append(f"const mod = require('{module_path}');\n")
            for fn in functions:
                lines.append(f"test('{fn['name']} works', () => {{")
                lines.append(f"  expect(mod.{fn['name']}()).toBeDefined();")
                lines.append(f"}});\n")
        return "\n".join(lines) + "\n"

    async def generate_tests(self, kwargs_dict):
        """Analyze a source file and generate a test skeleton."""
        kwargs = kwargs_dict or {}
        path_value = str(kwargs.get("path", "")).strip()
        output_path = str(kwargs.get("output_path", "")).strip()
        dry_run = bool(kwargs.get("dry_run", True))

        if not path_value:
            return self._json_response({"status": "failed", "error": "path is required"})

        try:
            source_path = self.resolve_workspace_path(path_value, must_exist=True)
        except Exception as e:
            return self._json_response({"status": "failed", "error": str(e)})

        if not source_path.is_file():
            return self._json_response({"status": "failed", "error": f"Not a file: {source_path}"})

        content = source_path.read_text(encoding="utf-8", errors="replace")
        language = self._detect_language(source_path)
        if language == "unknown":
            return self._json_response({"status": "failed", "error": f"Unsupported language for {source_path.suffix}"})

        framework_info = self._detect_test_framework(language)
        functions = self._extract_functions_for_testing(content, language)

        if not functions:
            return self._json_response({"status": "ok", "message": "No testable functions found", "path": path_value})

        if language == "python":
            test_code = self._generate_python_test_skeleton(source_path, functions, framework_info["framework"])
            default_output = source_path.parent / "tests" / f"test_{source_path.stem}.py"
        else:
            test_code = self._generate_js_test_skeleton(source_path, functions, framework_info["framework"])
            default_output = source_path.parent / "tests" / f"{source_path.stem}.test.js"

        if output_path:
            try:
                target = self.resolve_workspace_path(output_path, must_exist=False)
            except Exception:
                target = default_output
        else:
            target = default_output

        wrote = False
        if not dry_run:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(test_code, encoding="utf-8")
            wrote = True

        return self._json_response({
            "status": "ok",
            "source_path": source_path.relative_to(self.workspace_root).as_posix(),
            "language": language,
            "framework": framework_info["framework"],
            "functions_found": len(functions),
            "functions": [{"name": f["name"], "line": f.get("line", 0)} for f in functions],
            "test_output_path": target.relative_to(self.workspace_root).as_posix(),
            "test_code": test_code if dry_run else "(written to file)",
            "dry_run": dry_run,
            "wrote": wrote,
        })

    async def run_tests(self, kwargs_dict=None):
        """Run test suite and return structured results."""
        kwargs = kwargs_dict or {}
        path_value = str(kwargs.get("path", ".")).strip()
        framework = str(kwargs.get("framework", "")).strip()
        timeout_sec = float(kwargs.get("timeout_sec", 60))
        pattern = str(kwargs.get("pattern", "")).strip()

        try:
            test_root = self.resolve_workspace_path(path_value, must_exist=True)
        except Exception as e:
            return self._json_response({"status": "failed", "error": str(e)})

        if not framework:
            # Auto-detect
            if test_root.suffix == ".py" or any(test_root.rglob("test_*.py")):
                framework_info = self._detect_test_framework("python")
            elif test_root.suffix in {".js", ".ts"} or any(test_root.rglob("*.test.js")):
                framework_info = self._detect_test_framework("javascript")
            else:
                framework_info = self._detect_test_framework("python")
            framework = framework_info["framework"]
            runner = framework_info["runner"]
        else:
            runner = {"pytest": "python -m pytest", "unittest": "python -m unittest", "jest": "npx jest", "vitest": "npx vitest", "node:test": "node --test"}.get(framework, framework)

        # Build command
        if framework in {"pytest", "unittest"}:
            cmd = f"{runner} discover -s {test_root} -p 'test_*.py' -v" if framework == "unittest" else f"{runner} {test_root} -v"
        elif framework == "node:test":
            target = f"{test_root}/*.test.js" if test_root.is_dir() else str(test_root)
            cmd = f"{runner} {target}"
        else:
            cmd = f"{runner} {test_root}"

        if pattern:
            cmd += f" -k {pattern}" if framework in {"pytest", "unittest"} else f" --grep {pattern}"

        try:
            proc = subprocess.run(
                cmd, shell=True,
                cwd=str(self.workspace_root),
                capture_output=True, text=True,
                timeout=max(5.0, min(timeout_sec, 300.0)),
            )
            output = proc.stdout + proc.stderr
            passed = re.findall(r"(\d+)\s+(?:passed|pass)", output, re.IGNORECASE)
            failed = re.findall(r"(\d+)\s+(?:failed|fail)", output, re.IGNORECASE)
            errors = re.findall(r"(\d+)\s+error", output, re.IGNORECASE)

            return self._json_response({
                "status": "ok" if proc.returncode == 0 else "failed",
                "framework": framework,
                "exit_code": proc.returncode,
                "passed": int(passed[0]) if passed else 0,
                "failed": int(failed[0]) if failed else 0,
                "errors": int(errors[0]) if errors else 0,
                "output": _redact_sensitive_text(output[:15000], max_chars=15000),
                "truncated": len(output) > 15000,
            })
        except subprocess.TimeoutExpired:
            return self._json_response({"status": "failed", "error": f"Tests timed out after {timeout_sec}s"})
        except Exception as e:
            return self._json_response({"status": "failed", "error": str(e)})

    async def coverage_gaps(self, kwargs_dict=None):
        """Analyze test coverage gaps for a source file."""
        kwargs = kwargs_dict or {}
        path_value = str(kwargs.get("path", "")).strip()

        if not path_value:
            return self._json_response({"status": "failed", "error": "path is required"})

        try:
            source_path = self.resolve_workspace_path(path_value, must_exist=True)
        except Exception as e:
            return self._json_response({"status": "failed", "error": str(e)})

        content = source_path.read_text(encoding="utf-8", errors="replace")
        language = self._detect_language(source_path)
        functions = self._extract_functions_for_testing(content, language)
        function_names = {f["name"] for f in functions}

        # Find existing test files
        test_dir = source_path.parent / "tests"
        tested_names = set()
        if test_dir.is_dir():
            for test_file in test_dir.rglob("test_*.*"):
                try:
                    test_content = test_file.read_text(encoding="utf-8", errors="replace")
                    for name in function_names:
                        if name in test_content:
                            tested_names.add(name)
                except Exception:
                    continue

        untested = sorted(function_names - tested_names)
        tested = sorted(tested_names)
        coverage_pct = round(len(tested) / max(1, len(function_names)) * 100, 1)

        return self._json_response({
            "status": "ok",
            "path": source_path.relative_to(self.workspace_root).as_posix(),
            "total_functions": len(function_names),
            "tested": len(tested),
            "untested": len(untested),
            "coverage_percent": coverage_pct,
            "tested_names": tested,
            "untested_names": untested,
        })
