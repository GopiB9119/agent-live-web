import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from memory_tools import MemoryManager


class MemoryManagerTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.workspace = Path(self.tmp.name).resolve()
        self.memory_dir = self.workspace / "memory"
        self.long_term = self.workspace / "MEMORY.md"
        self.vector_index = self.memory_dir / ".vector_index.json"

        def _resolve_workspace_path(raw_path, must_exist=False):
            candidate = Path(raw_path)
            if not candidate.is_absolute():
                candidate = (self.workspace / candidate).resolve()
            else:
                candidate = candidate.resolve()
            if self.workspace not in candidate.parents and candidate != self.workspace:
                raise ValueError("outside workspace")
            if must_exist and not candidate.exists():
                raise FileNotFoundError(str(candidate))
            return candidate

        def _redact(value):
            return str(value).replace("SECRET", "[REDACTED]")

        self.manager = MemoryManager(
            workspace_root=self.workspace,
            memory_dir=self.memory_dir,
            long_term_memory_file=self.long_term,
            vector_index_file=self.vector_index,
            vector_dim=64,
            resolve_workspace_path_fn=_resolve_workspace_path,
            redact_sensitive_text_fn=_redact,
        )

    def tearDown(self):
        self.tmp.cleanup()

    async def test_memory_log_and_get(self):
        logged = json.loads(
            await self.manager.memory_log(
                {
                    "content": "Use SECRET token for api",
                    "role": "user",
                    "importance": 9,
                    "tags": ["security", "note"],
                }
            )
        )
        self.assertEqual(logged["status"], "ok")

        fetched = json.loads(await self.manager.memory_get({"file": logged["file"], "start_line": 1, "end_line": 80}))
        self.assertEqual(fetched["status"], "ok")
        self.assertIn("[REDACTED]", fetched["content"])
        self.assertIn("role:user", fetched["content"])

    async def test_memory_promote_and_bootstrap(self):
        promoted = json.loads(await self.manager.memory_promote({"fact": "Playwright is primary tool", "importance": 8}))
        self.assertEqual(promoted["status"], "ok")
        self.assertTrue(self.long_term.exists())

        await self.manager.memory_log({"content": "Today tested playwright edge", "importance": 5})
        boot = json.loads(await self.manager.memory_bootstrap({"include_long_term": True}))
        self.assertEqual(boot["status"], "ok")
        self.assertTrue(any(path.endswith("MEMORY.md") for path in boot["files"]))

    async def test_memory_search_and_reindex(self):
        await self.manager.memory_log({"content": "Playwright edge mcp stable verification run", "importance": 7})
        await self.manager.memory_promote({"fact": "Use local-first security model", "importance": 9})

        reindex = json.loads(await self.manager.memory_reindex({"days_back": 7, "include_long_term": True}))
        self.assertEqual(reindex["status"], "ok")
        self.assertTrue(self.vector_index.exists())
        self.assertGreaterEqual(int(reindex["items"]), 1)

        search = json.loads(
            await self.manager.memory_search(
                {
                    "query": "playwright security model",
                    "top_k": 5,
                    "days_back": 7,
                    "include_long_term": True,
                    "use_semantic": True,
                }
            )
        )
        self.assertEqual(search["status"], "ok")
        self.assertGreaterEqual(search["count"], 1)


if __name__ == "__main__":
    unittest.main()
