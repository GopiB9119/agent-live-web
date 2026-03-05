import json
import os
import re
import subprocess


class CommandManager:
    """
    Workspace-scoped command execution with restricted/permissive safety controls.
    """

    def __init__(
        self,
        workspace_root,
        resolve_workspace_path_fn,
        run_command_security_mode_default: str,
        run_command_allow_dangerous_env: str,
        to_bool_fn,
        run_command_is_safe_in_restricted_mode_fn,
    ):
        self.workspace_root = workspace_root
        self.resolve_workspace_path = resolve_workspace_path_fn
        self.run_command_security_mode_default = str(run_command_security_mode_default or "restricted").strip().lower() or "restricted"
        self.run_command_allow_dangerous_env = str(run_command_allow_dangerous_env or "AGENT_ALLOW_DANGEROUS_COMMANDS")
        self.to_bool = to_bool_fn
        self.run_command_is_safe_in_restricted_mode = run_command_is_safe_in_restricted_mode_fn

    async def run_command(self, kwargs_dict):
        kwargs = kwargs_dict or {}
        command = str(kwargs.get("command", "")).strip()
        cwd_value = kwargs.get("cwd", ".")
        timeout_sec = float(kwargs.get("timeout_sec", 30))
        security_mode = str(kwargs.get("security_mode", self.run_command_security_mode_default)).strip().lower() or self.run_command_security_mode_default
        confirm = self.to_bool(kwargs.get("confirm", False), False)
        allow_dangerous = bool(kwargs.get("allow_dangerous", False))

        if not command:
            return json.dumps({"status": "failed", "error": "command is required"}, ensure_ascii=True)
        if security_mode not in {"restricted", "permissive"}:
            return json.dumps({"status": "failed", "error": "security_mode must be 'restricted' or 'permissive'"}, ensure_ascii=True)

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

        if security_mode == "restricted" and not allow_dangerous:
            if not self.run_command_is_safe_in_restricted_mode(command):
                return json.dumps(
                    {
                        "status": "blocked",
                        "error": "Command blocked by restricted security mode. Use a safe command pattern or switch to permissive mode explicitly.",
                        "command": command,
                        "security_mode": security_mode,
                    },
                    ensure_ascii=True,
                )

        if allow_dangerous:
            if not confirm:
                return json.dumps(
                    {
                        "status": "blocked",
                        "error": "allow_dangerous=true requires confirm=true in the same call.",
                        "command": command,
                    },
                    ensure_ascii=True,
                )
            if not self.to_bool(os.getenv(self.run_command_allow_dangerous_env, "0"), False):
                return json.dumps(
                    {
                        "status": "blocked",
                        "error": f"allow_dangerous=true requires {self.run_command_allow_dangerous_env}=1 in environment.",
                        "command": command,
                    },
                    ensure_ascii=True,
                )

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
            cwd_path = self.resolve_workspace_path(cwd_value, must_exist=True)
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
                    "security_mode": security_mode,
                    "cwd": cwd_path.relative_to(self.workspace_root).as_posix() if cwd_path != self.workspace_root else ".",
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
