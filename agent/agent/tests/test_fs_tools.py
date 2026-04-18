import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fs_tools import FSManager


class FSManagerTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

        def _resolve(raw_path, must_exist=False):
            candidate = Path(raw_path)
            if not candidate.is_absolute():
                candidate = (self.root / candidate).resolve()
            else:
                candidate = candidate.resolve()
            candidate.relative_to(self.root)
            if must_exist and not candidate.exists():
                raise FileNotFoundError(str(candidate))
            return candidate

        self.manager = FSManager(
            workspace_root=self.root,
            resolve_workspace_path_fn=_resolve,
            noise_dir_names={".git", "node_modules", "__pycache__", ".venv"},
            binary_suffixes={".png", ".jpg", ".zip", ".exe"},
        )

    def tearDown(self):
        self.tmp.cleanup()

    async def test_fs_read_redacts_secret_bearing_content(self):
        target = self.root / ".env"
        target.write_text("OPENAI_API_KEY=abc123\nAuthorization=secret-value\nSAFE=ok\n", encoding="utf-8")

        raw = await self.manager.fs_read({"path": ".env"})
        result = json.loads(raw)

        self.assertEqual(result["status"], "ok")
        self.assertIn("OPENAI_API_KEY=[REDACTED]", result["content"])
        self.assertIn("Authorization=[REDACTED]", result["content"])
        self.assertIn("SAFE=ok", result["content"])
        self.assertNotIn("abc123", raw)

    async def test_fs_read_batch_redacts_each_file_content(self):
        (self.root / ".env").write_text("token=abc123\n", encoding="utf-8")
        (self.root / "config.txt").write_text("cookie=sessionid=123\nname=value\n", encoding="utf-8")

        raw = await self.manager.fs_read_batch({"paths": [".env", "config.txt"]})
        result = json.loads(raw)

        self.assertEqual(result["status"], "ok")
        self.assertIn("token=[REDACTED]", result["results"][0]["content"])
        self.assertIn("cookie=[REDACTED]", result["results"][1]["content"])
        self.assertIn("name=value", result["results"][1]["content"])
        self.assertNotIn("abc123", raw)

    async def test_fs_search_redacts_matching_line_text(self):
        target = self.root / "secrets.txt"
        target.write_text("prefix token=abc123 suffix\n", encoding="utf-8")

        raw = await self.manager.fs_search({"path": ".", "pattern": "token", "file_glob": "*.txt"})
        result = json.loads(raw)

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["count"], 1)
        self.assertIn("token=[REDACTED]", result["matches"][0]["text"])
        self.assertNotIn("abc123", raw)

    async def test_fs_analyze_file_redacts_preview_only(self):
        target = self.root / "module.py"
        target.write_text(
            "API_KEY = 'abc123'\n"
            "def hello():\n"
            "    return 'world'\n",
            encoding="utf-8",
        )

        raw = await self.manager.fs_analyze_file({"path": "module.py", "include_preview": True})
        result = json.loads(raw)

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["analysis"]["language"], "python")
        self.assertIn("API_KEY=[REDACTED]", result["preview"])
        self.assertIn("hello", result["analysis"]["functions"])
        self.assertNotIn("abc123", raw)
