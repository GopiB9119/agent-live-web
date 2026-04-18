import json
import os
import re
import subprocess
from pathlib import Path
try:
    from runtime_utils import redact_sensitive_data as _redact_sensitive_data, redact_sensitive_text as _redact_sensitive_text
except Exception:
    from .runtime_utils import redact_sensitive_data as _redact_sensitive_data, redact_sensitive_text as _redact_sensitive_text


class GitManager:
    """
    Git operations manager: status, diff, log, blame, commit, branch, stash.
    Workspace-scoped, safe by default. Destructive ops require explicit confirm.
    """

    def __init__(self, workspace_root: Path, resolve_workspace_path_fn):
        self.workspace_root = Path(workspace_root).resolve()
        self.resolve_workspace_path = resolve_workspace_path_fn

    @staticmethod
    def _json_response(payload, max_chars=20000):
        return json.dumps(_redact_sensitive_data(payload, max_chars=max_chars), ensure_ascii=True)

    def _run_git(self, args: list, cwd=None, timeout=30) -> dict:
        """Run a git command and return structured result."""
        max_stdout_chars = 20000
        try:
            proc = subprocess.run(
                ["git"] + args,
                cwd=str(cwd or self.workspace_root),
                capture_output=True,
                text=True,
                timeout=max(1.0, min(timeout, 120.0)),
            )
            return {
                "ok": proc.returncode == 0,
                "exit_code": proc.returncode,
                "stdout": proc.stdout[:max_stdout_chars],
                "stderr": proc.stderr[:8000],
            }
        except subprocess.TimeoutExpired:
            return {"ok": False, "exit_code": -1, "stdout": "", "stderr": "Git command timed out."}
        except FileNotFoundError:
            return {"ok": False, "exit_code": -1, "stdout": "", "stderr": "Git is not installed or not in PATH."}
        except Exception as e:
            return {"ok": False, "exit_code": -1, "stdout": "", "stderr": str(e)}

    async def git_status(self, kwargs_dict=None):
        """Get working tree status: modified, staged, untracked files."""
        result = self._run_git(["status", "--porcelain=v1", "-uall"])
        if not result["ok"]:
            return self._json_response({"status": "failed", "error": result["stderr"]})

        files = {"modified": [], "staged": [], "untracked": [], "deleted": []}
        for line in result["stdout"].splitlines():
            if len(line) < 4:
                continue
            index_status = line[0]
            work_status = line[1]
            filepath = line[3:].strip()
            if index_status == "?" and work_status == "?":
                files["untracked"].append(filepath)
            elif index_status in {"A", "M", "R", "C"}:
                files["staged"].append(filepath)
            if work_status == "M":
                files["modified"].append(filepath)
            elif work_status == "D":
                files["deleted"].append(filepath)

        branch_result = self._run_git(["branch", "--show-current"])
        branch = branch_result["stdout"].strip() if branch_result["ok"] else "unknown"

        return self._json_response({
            "status": "ok",
            "branch": branch,
            "files": files,
            "total_changes": sum(len(v) for v in files.values()),
            "clean": all(len(v) == 0 for v in files.values()),
        })

    async def git_diff(self, kwargs_dict=None):
        """Show diff of working tree or staged changes."""
        kwargs = kwargs_dict or {}
        staged = bool(kwargs.get("staged", False))
        path_filter = str(kwargs.get("path", "")).strip()
        max_chars = int(kwargs.get("max_chars", 15000))
        max_chars = max(500, min(max_chars, 100000))

        args = ["diff", "--stat"]
        if staged:
            args.append("--cached")
        if path_filter:
            try:
                resolved = self.resolve_workspace_path(path_filter, must_exist=True)
                args.extend(["--", str(resolved)])
            except Exception:
                args.extend(["--", path_filter])

        stat_result = self._run_git(args)

        detail_args = ["diff"]
        if staged:
            detail_args.append("--cached")
        if path_filter:
            detail_args.extend(["--", path_filter])
        detail_result = self._run_git(detail_args)

        diff_text = detail_result["stdout"][:max_chars]
        truncated = len(detail_result["stdout"]) > max_chars

        return self._json_response({
            "status": "ok" if detail_result["ok"] else "failed",
            "staged": staged,
            "stat": stat_result["stdout"][:4000],
            "diff": _redact_sensitive_text(diff_text, max_chars=max_chars),
            "truncated": truncated,
            "error": detail_result["stderr"] if not detail_result["ok"] else None,
        })

    async def git_log(self, kwargs_dict=None):
        """Show recent commit history."""
        kwargs = kwargs_dict or {}
        count = max(1, min(int(kwargs.get("count", 10)), 50))
        oneline = bool(kwargs.get("oneline", True))
        path_filter = str(kwargs.get("path", "")).strip()

        args = ["log", f"-{count}"]
        if oneline:
            args.append("--oneline")
        else:
            args.extend(["--format=%H|%an|%ae|%ai|%s"])
        if path_filter:
            args.extend(["--", path_filter])

        result = self._run_git(args)
        if not result["ok"]:
            return self._json_response({"status": "failed", "error": result["stderr"]})

        commits = []
        for line in result["stdout"].strip().splitlines():
            if oneline:
                parts = line.split(" ", 1)
                commits.append({"hash": parts[0], "message": parts[1] if len(parts) > 1 else ""})
            else:
                parts = line.split("|", 4)
                if len(parts) >= 5:
                    commits.append({
                        "hash": parts[0],
                        "author": parts[1],
                        "email": _redact_sensitive_text(parts[2], max_chars=200),
                        "date": parts[3],
                        "message": parts[4],
                    })

        return self._json_response({"status": "ok", "count": len(commits), "commits": commits})

    async def git_blame(self, kwargs_dict=None):
        """Show line-by-line authorship for a file."""
        kwargs = kwargs_dict or {}
        path_value = str(kwargs.get("path", "")).strip()
        start_line = int(kwargs.get("start_line", 0))
        end_line = int(kwargs.get("end_line", 0))

        if not path_value:
            return self._json_response({"status": "failed", "error": "path is required"})

        args = ["blame", "--porcelain"]
        if start_line > 0 and end_line >= start_line:
            args.extend([f"-L{start_line},{end_line}"])
        args.append(path_value)

        result = self._run_git(args)
        if not result["ok"]:
            return self._json_response({"status": "failed", "error": result["stderr"], "path": path_value})

        # Parse porcelain blame into structured entries
        entries = []
        current = {}
        for line in result["stdout"].splitlines():
            if line.startswith("\t"):
                current["content"] = line[1:]
                entries.append(current)
                current = {}
            elif re.match(r"^[0-9a-f]{40}\s", line):
                parts = line.split()
                current["hash"] = parts[0]
                current["original_line"] = int(parts[1]) if len(parts) > 1 else 0
                current["final_line"] = int(parts[2]) if len(parts) > 2 else 0
            elif line.startswith("author "):
                current["author"] = line[7:]
            elif line.startswith("author-time "):
                current["timestamp"] = line[12:]

        return self._json_response({
            "status": "ok",
            "path": path_value,
            "entries": entries[:500],
            "count": len(entries),
        })

    async def git_commit(self, kwargs_dict=None):
        """Stage and commit changes. Requires explicit confirm=true."""
        kwargs = kwargs_dict or {}
        message = str(kwargs.get("message", "")).strip()
        paths = kwargs.get("paths", [])
        stage_all = bool(kwargs.get("stage_all", False))
        confirm = bool(kwargs.get("confirm", False))

        if not message:
            return self._json_response({"status": "failed", "error": "commit message is required"})
        if not confirm:
            return self._json_response({
                "status": "blocked",
                "error": "git_commit requires confirm=true. Review changes with git_diff first.",
                "message": message,
            })

        # Stage files
        if stage_all:
            stage_result = self._run_git(["add", "-A"])
        elif isinstance(paths, list) and paths:
            stage_result = self._run_git(["add", "--"] + [str(p) for p in paths])
        else:
            return self._json_response({"status": "failed", "error": "Provide paths or set stage_all=true"})

        if not stage_result["ok"]:
            return self._json_response({"status": "failed", "error": f"Stage failed: {stage_result['stderr']}"})

        # Commit
        commit_result = self._run_git(["commit", "-m", message])
        if not commit_result["ok"]:
            return self._json_response({"status": "failed", "error": commit_result["stderr"]})

        # Get the new commit hash
        hash_result = self._run_git(["rev-parse", "HEAD"])
        commit_hash = hash_result["stdout"].strip() if hash_result["ok"] else "unknown"

        return self._json_response({
            "status": "ok",
            "message": message,
            "hash": commit_hash,
            "output": commit_result["stdout"][:4000],
        })

    async def git_branch(self, kwargs_dict=None):
        """List, create, switch, or delete branches."""
        kwargs = kwargs_dict or {}
        action = str(kwargs.get("action", "list")).strip().lower()
        name = str(kwargs.get("name", "")).strip()

        if action == "list":
            result = self._run_git(["branch", "-a", "--format=%(refname:short) %(objectname:short) %(upstream:short)"])
            if not result["ok"]:
                return self._json_response({"status": "failed", "error": result["stderr"]})
            current_result = self._run_git(["branch", "--show-current"])
            current = current_result["stdout"].strip() if current_result["ok"] else ""
            branches = []
            for line in result["stdout"].strip().splitlines():
                parts = line.split()
                branch_name = parts[0] if parts else ""
                branches.append({
                    "name": branch_name,
                    "hash": parts[1] if len(parts) > 1 else "",
                    "upstream": parts[2] if len(parts) > 2 else "",
                    "current": branch_name == current,
                })
            return self._json_response({"status": "ok", "current": current, "branches": branches})

        if not name:
            return self._json_response({"status": "failed", "error": "branch name is required"})

        if action == "create":
            result = self._run_git(["checkout", "-b", name])
        elif action == "switch":
            result = self._run_git(["checkout", name])
        elif action == "delete":
            confirm = bool(kwargs.get("confirm", False))
            if not confirm:
                return self._json_response({"status": "blocked", "error": "branch delete requires confirm=true"})
            result = self._run_git(["branch", "-d", name])
        else:
            return self._json_response({"status": "failed", "error": f"Unknown action: {action}. Use list/create/switch/delete."})

        return self._json_response({
            "status": "ok" if result["ok"] else "failed",
            "action": action,
            "name": name,
            "output": result["stdout"][:4000],
            "error": result["stderr"] if not result["ok"] else None,
        })

    async def git_stash(self, kwargs_dict=None):
        """Save or restore working changes via git stash."""
        kwargs = kwargs_dict or {}
        action = str(kwargs.get("action", "push")).strip().lower()
        message = str(kwargs.get("message", "")).strip()

        if action == "push":
            args = ["stash", "push"]
            if message:
                args.extend(["-m", message])
            result = self._run_git(args)
        elif action == "pop":
            result = self._run_git(["stash", "pop"])
        elif action == "list":
            result = self._run_git(["stash", "list"])
        elif action == "drop":
            confirm = bool(kwargs.get("confirm", False))
            if not confirm:
                return self._json_response({"status": "blocked", "error": "stash drop requires confirm=true"})
            result = self._run_git(["stash", "drop"])
        else:
            return self._json_response({"status": "failed", "error": f"Unknown stash action: {action}"})

        return self._json_response({
            "status": "ok" if result["ok"] else "failed",
            "action": action,
            "output": result["stdout"][:4000],
            "error": result["stderr"] if not result["ok"] else None,
        })
