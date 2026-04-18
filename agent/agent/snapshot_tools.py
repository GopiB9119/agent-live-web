import json
import hashlib
import shutil
from datetime import datetime
from pathlib import Path
try:
    from runtime_utils import redact_sensitive_data as _redact_sensitive_data
except Exception:
    from .runtime_utils import redact_sensitive_data as _redact_sensitive_data


class SnapshotManager:
    """
    File snapshot and rollback manager.
    Takes snapshots before edits so changes can be undone.
    """

    SNAPSHOT_DIR_NAME = ".agent-snapshots"

    def __init__(self, workspace_root: Path, resolve_workspace_path_fn, max_snapshots: int = 50):
        self.workspace_root = Path(workspace_root).resolve()
        self.resolve_workspace_path = resolve_workspace_path_fn
        self.max_snapshots = max(5, min(max_snapshots, 200))
        self.snapshot_dir = self.workspace_root / self.SNAPSHOT_DIR_NAME

    @staticmethod
    def _json_response(payload, max_chars=20000):
        return json.dumps(_redact_sensitive_data(payload, max_chars=max_chars), ensure_ascii=True)

    def _ensure_snapshot_dir(self):
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)

    def _snapshot_id(self) -> str:
        return datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:21]

    def _metadata_path(self, snapshot_id: str) -> Path:
        return self.snapshot_dir / f"{snapshot_id}.json"

    def _content_path(self, snapshot_id: str, rel_path: str) -> Path:
        safe_name = hashlib.sha1(rel_path.encode("utf-8")).hexdigest()[:16]
        return self.snapshot_dir / f"{snapshot_id}_{safe_name}"

    def _prune_old_snapshots(self):
        """Keep only the most recent max_snapshots."""
        meta_files = sorted(self.snapshot_dir.glob("*.json"), key=lambda p: p.name, reverse=True)
        for old_meta in meta_files[self.max_snapshots:]:
            try:
                meta = json.loads(old_meta.read_text(encoding="utf-8"))
                for file_entry in meta.get("files", []):
                    content_path = self.snapshot_dir / file_entry.get("backup_name", "")
                    if content_path.exists():
                        content_path.unlink(missing_ok=True)
                old_meta.unlink(missing_ok=True)
            except Exception:
                pass

    async def snapshot_create(self, kwargs_dict=None):
        """Create a snapshot of one or more files before editing."""
        kwargs = kwargs_dict or {}
        paths = kwargs.get("paths", [])
        label = str(kwargs.get("label", "")).strip() or "manual snapshot"

        if not isinstance(paths, list) or not paths:
            return self._json_response({"status": "failed", "error": "paths must be a non-empty array"})

        self._ensure_snapshot_dir()
        snapshot_id = self._snapshot_id()
        file_entries = []

        for raw_path in paths[:50]:
            try:
                file_path = self.resolve_workspace_path(raw_path, must_exist=True)
                if not file_path.is_file():
                    continue
                rel = file_path.relative_to(self.workspace_root).as_posix()
                content = file_path.read_bytes()
                content_hash = hashlib.sha1(content).hexdigest()
                backup_name = f"{snapshot_id}_{hashlib.sha1(rel.encode()).hexdigest()[:16]}"
                backup_path = self.snapshot_dir / backup_name
                backup_path.write_bytes(content)
                file_entries.append({
                    "path": rel,
                    "backup_name": backup_name,
                    "size": len(content),
                    "hash": content_hash,
                })
            except Exception as e:
                file_entries.append({"path": str(raw_path), "error": str(e)})

        metadata = {
            "snapshot_id": snapshot_id,
            "label": label,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "files": file_entries,
        }
        self._metadata_path(snapshot_id).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        self._prune_old_snapshots()

        return self._json_response({
            "status": "ok",
            "snapshot_id": snapshot_id,
            "label": label,
            "files_saved": len([f for f in file_entries if "error" not in f]),
            "files": [{"path": f["path"], "size": f.get("size", 0)} for f in file_entries if "error" not in f],
        })

    async def snapshot_restore(self, kwargs_dict=None):
        """Restore files from a snapshot (undo edits)."""
        kwargs = kwargs_dict or {}
        snapshot_id = str(kwargs.get("snapshot_id", "")).strip()
        confirm = bool(kwargs.get("confirm", False))

        if not snapshot_id:
            return self._json_response({"status": "failed", "error": "snapshot_id is required"})
        if not confirm:
            return self._json_response({"status": "blocked", "error": "snapshot_restore requires confirm=true"})

        meta_path = self._metadata_path(snapshot_id)
        if not meta_path.exists():
            return self._json_response({"status": "failed", "error": f"Snapshot not found: {snapshot_id}"})

        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        restored = []
        errors = []

        for file_entry in metadata.get("files", []):
            backup_name = file_entry.get("backup_name", "")
            rel_path = file_entry.get("path", "")
            backup_path = self.snapshot_dir / backup_name
            if not backup_path.exists():
                errors.append({"path": rel_path, "error": "Backup file missing"})
                continue
            try:
                target = self.workspace_root / rel_path
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(backup_path), str(target))
                restored.append(rel_path)
            except Exception as e:
                errors.append({"path": rel_path, "error": str(e)})

        return self._json_response({
            "status": "ok" if not errors else "partial",
            "snapshot_id": snapshot_id,
            "restored": len(restored),
            "restored_files": restored,
            "errors": errors,
        })

    async def snapshot_list(self, kwargs_dict=None):
        """List available snapshots."""
        kwargs = kwargs_dict or {}
        limit = max(1, min(int(kwargs.get("limit", 20)), 100))

        self._ensure_snapshot_dir()
        meta_files = sorted(self.snapshot_dir.glob("*.json"), key=lambda p: p.name, reverse=True)

        snapshots = []
        for meta_path in meta_files[:limit]:
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                snapshots.append({
                    "snapshot_id": meta.get("snapshot_id", ""),
                    "label": meta.get("label", ""),
                    "created_at": meta.get("created_at", ""),
                    "file_count": len(meta.get("files", [])),
                })
            except Exception:
                continue

        return self._json_response({"status": "ok", "count": len(snapshots), "snapshots": snapshots})

    async def snapshot_diff(self, kwargs_dict=None):
        """Show what changed since a snapshot was taken."""
        kwargs = kwargs_dict or {}
        snapshot_id = str(kwargs.get("snapshot_id", "")).strip()

        if not snapshot_id:
            return self._json_response({"status": "failed", "error": "snapshot_id is required"})

        meta_path = self._metadata_path(snapshot_id)
        if not meta_path.exists():
            return self._json_response({"status": "failed", "error": f"Snapshot not found: {snapshot_id}"})

        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        changes = []

        for file_entry in metadata.get("files", []):
            rel_path = file_entry.get("path", "")
            original_hash = file_entry.get("hash", "")
            target = self.workspace_root / rel_path
            if not target.exists():
                changes.append({"path": rel_path, "change": "deleted"})
            else:
                current_hash = hashlib.sha1(target.read_bytes()).hexdigest()
                if current_hash != original_hash:
                    changes.append({"path": rel_path, "change": "modified"})
                else:
                    changes.append({"path": rel_path, "change": "unchanged"})

        return self._json_response({
            "status": "ok",
            "snapshot_id": snapshot_id,
            "changes": changes,
            "modified_count": sum(1 for c in changes if c["change"] == "modified"),
            "deleted_count": sum(1 for c in changes if c["change"] == "deleted"),
        })
