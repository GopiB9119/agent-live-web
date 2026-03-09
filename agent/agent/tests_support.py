import shutil
import uuid
from pathlib import Path


def _create_named_dir(root: Path, prefix: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    temp_dir = (root / f"{prefix}-{uuid.uuid4().hex}").resolve()
    temp_dir.mkdir(parents=True, exist_ok=False)
    return temp_dir


def create_repo_local_temp_dir(test_file: Path, bucket: str, prefix: str) -> Path:
    """
    Create a writable repo-local temp directory for tests.

    Windows sandboxed environments in this project have proven unreliable with
    tempfile-based directories, even when they live under the workspace. Use a
    normal mkdir-based path instead so tests exercise product code, not host
    temp-directory permissions.
    """

    root = Path(test_file).resolve().parents[2] / ".agent-state" / bucket
    return _create_named_dir(root, prefix)


def create_temp_dir_under(root: Path, prefix: str) -> Path:
    return _create_named_dir(Path(root).resolve(), prefix)


def remove_tree(path_obj: Path) -> None:
    shutil.rmtree(path_obj, ignore_errors=True)
