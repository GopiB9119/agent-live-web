import json
import hashlib
import math
import re
from datetime import datetime, timedelta
from pathlib import Path


class MemoryManager:
    """
    Markdown + vector-style memory manager for the Python agent.
    """

    def __init__(
        self,
        workspace_root: Path,
        memory_dir: Path,
        long_term_memory_file: Path,
        vector_index_file: Path,
        vector_dim: int,
        resolve_workspace_path_fn,
        redact_sensitive_text_fn=None,
    ):
        self.workspace_root = Path(workspace_root).resolve()
        self.memory_dir = Path(memory_dir).resolve()
        self.long_term_memory_file = Path(long_term_memory_file).resolve()
        self.vector_index_file = Path(vector_index_file).resolve()
        self.vector_dim = int(vector_dim)
        self.resolve_workspace_path = resolve_workspace_path_fn
        self.redact_sensitive_text = redact_sensitive_text_fn

    def _ensure_memory_paths(self):
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        if not self.long_term_memory_file.exists():
            self.long_term_memory_file.write_text("# Curated Long-Term Memory\n\n", encoding="utf-8")

    def _daily_memory_file(self, target_date: datetime) -> Path:
        self._ensure_memory_paths()
        return self.memory_dir / f"{target_date.strftime('%Y-%m-%d')}.md"

    def _append_daily_memory(self, content: str, role: str = "event", importance: int = 3, tags=None):
        tags = tags or []
        self._ensure_memory_paths()
        now = datetime.now()
        file_path = self._daily_memory_file(now)
        if not file_path.exists():
            file_path.write_text(f"# Daily Memory Log - {now.strftime('%Y-%m-%d')}\n\n", encoding="utf-8")
        safe_importance = max(1, min(int(importance), 10))
        tag_text = ",".join(sorted({str(tag).strip() for tag in tags if str(tag).strip()}))
        header = f"## {now.isoformat(timespec='seconds')} | role:{role} | importance:{safe_importance}"
        if tag_text:
            header += f" | tags:{tag_text}"
        body = str(content).strip()
        entry = f"{header}\n- {body}\n\n"
        with file_path.open("a", encoding="utf-8", errors="replace", newline="") as fh:
            fh.write(entry)
        return file_path

    @staticmethod
    def _read_file_lines(path_obj: Path):
        text = path_obj.read_text(encoding="utf-8", errors="replace")
        return text, text.splitlines()

    @staticmethod
    def _tokenize_for_memory(text: str):
        return re.findall(r"[a-zA-Z0-9_]+", text.lower())

    def _memory_chunk_score(self, query_tokens, chunk_text: str, recency_boost: float = 0.0):
        chunk_tokens = self._tokenize_for_memory(chunk_text)
        if not chunk_tokens:
            return 0.0
        chunk_set = set(chunk_tokens)
        overlap = sum(1 for token in query_tokens if token in chunk_set)
        density = overlap / max(1, len(chunk_set))
        importance_match = re.search(r"importance:(\d+)", chunk_text)
        importance = int(importance_match.group(1)) if importance_match else 3
        importance_boost = min(importance, 10) * 0.15
        return overlap + density + importance_boost + recency_boost

    def _memory_file_recency_boost(self, path_obj: Path):
        if path_obj == self.long_term_memory_file:
            return 1.0
        try:
            date_str = path_obj.stem
            file_date = datetime.strptime(date_str, "%Y-%m-%d").date()
            today = datetime.now().date()
            if file_date == today:
                return 2.0
            if file_date == (today - timedelta(days=1)):
                return 1.5
            days_old = (today - file_date).days
            return max(0.1, 1.0 / (1 + days_old))
        except Exception:
            return 0.5

    @staticmethod
    def _memory_chunks_from_text(text: str, max_chunk_chars: int = 1600):
        sections = re.split(r"(?m)^##\s+", text)
        chunks = []
        for section in sections:
            section = section.strip()
            if not section:
                continue
            if len(section) <= max_chunk_chars:
                chunks.append(section)
                continue
            start = 0
            while start < len(section):
                chunks.append(section[start:start + max_chunk_chars])
                start += max_chunk_chars
        return chunks

    @staticmethod
    def _stable_token_hash(token: str) -> int:
        digest = hashlib.sha1(token.encode("utf-8", errors="ignore")).hexdigest()
        return int(digest[:8], 16)

    @staticmethod
    def _normalize_vector(values):
        if not values:
            return []
        magnitude = math.sqrt(sum(float(v) * float(v) for v in values))
        if magnitude <= 1e-12:
            return [0.0 for _ in values]
        return [float(v) / magnitude for v in values]

    def _hash_embed_text(self, text: str, dim: int = None):
        vector_dim = self.vector_dim if dim is None else max(16, int(dim))
        vector = [0.0] * vector_dim
        tokens = self._tokenize_for_memory(text)
        if not tokens:
            return vector
        for token in tokens:
            idx = self._stable_token_hash(token) % len(vector)
            vector[idx] += 1.0
            if len(token) > 5:
                idx2 = self._stable_token_hash(token[:5]) % len(vector)
                vector[idx2] += 0.5
        return self._normalize_vector(vector)

    @staticmethod
    def _cosine_similarity(vec_a, vec_b):
        if not vec_a or not vec_b:
            return 0.0
        size = min(len(vec_a), len(vec_b))
        if size == 0:
            return 0.0
        return float(sum(float(vec_a[i]) * float(vec_b[i]) for i in range(size)))

    @staticmethod
    def _memory_file_signature(path_obj: Path):
        try:
            stat = path_obj.stat()
            return f"{stat.st_size}:{getattr(stat, 'st_mtime_ns', int(stat.st_mtime * 1e9))}"
        except Exception:
            return "0:0"

    def _load_memory_vector_index(self):
        default_index = {
            "version": 1,
            "backend": "hash-embedding-v1",
            "dimension": self.vector_dim,
            "updated_at": "",
            "items": [],
        }
        try:
            if not self.vector_index_file.exists():
                return default_index
            data = json.loads(self.vector_index_file.read_text(encoding="utf-8", errors="replace"))
            if not isinstance(data, dict):
                return default_index
            if not isinstance(data.get("items"), list):
                data["items"] = []
            return data
        except Exception:
            return default_index

    def _save_memory_vector_index(self, index_data):
        self._ensure_memory_paths()
        index_data["updated_at"] = datetime.now().isoformat(timespec="seconds")
        self.vector_index_file.write_text(
            json.dumps(index_data, ensure_ascii=True),
            encoding="utf-8",
            errors="replace",
        )

    def _build_memory_vector_index(self, candidates, force_rebuild: bool = False, max_chunk_chars: int = 1600):
        index_data = self._load_memory_vector_index()
        existing = {}
        if not force_rebuild:
            for item in index_data.get("items", []):
                if not isinstance(item, dict):
                    continue
                item_id = str(item.get("id", "")).strip()
                if item_id:
                    existing[item_id] = item

        rebuilt_items = []
        touched_ids = set()

        for path_obj in candidates:
            if not path_obj.exists():
                continue
            rel = path_obj.relative_to(self.workspace_root).as_posix()
            signature = self._memory_file_signature(path_obj)
            recency = self._memory_file_recency_boost(path_obj)
            raw = path_obj.read_text(encoding="utf-8", errors="replace")
            chunks = self._memory_chunks_from_text(raw, max_chunk_chars=max_chunk_chars)

            for idx, chunk in enumerate(chunks, start=1):
                chunk_hash = hashlib.sha1(chunk.encode("utf-8", errors="ignore")).hexdigest()
                item_id = f"{rel}|{idx}|{chunk_hash[:16]}"
                touched_ids.add(item_id)

                cached = existing.get(item_id)
                if cached and isinstance(cached.get("vector"), list):
                    vector = cached["vector"]
                else:
                    vector = self._hash_embed_text(chunk, dim=self.vector_dim)

                rebuilt_items.append(
                    {
                        "id": item_id,
                        "file": rel,
                        "chunk_index": idx,
                        "chunk_hash": chunk_hash,
                        "file_signature": signature,
                        "recency_boost": float(recency),
                        "snippet": chunk[:1200],
                        "vector": vector,
                    }
                )

        index_data.update(
            {
                "version": 1,
                "backend": "hash-embedding-v1",
                "dimension": self.vector_dim,
                "items": rebuilt_items,
                "touched": len(touched_ids),
            }
        )
        self._save_memory_vector_index(index_data)
        return index_data

    async def memory_log(self, kwargs_dict):
        kwargs = kwargs_dict or {}
        content = str(kwargs.get("content", "")).strip()
        if self.redact_sensitive_text:
            content = self.redact_sensitive_text(content)
        role = str(kwargs.get("role", "event")).strip() or "event"
        importance = int(kwargs.get("importance", 3))
        tags = kwargs.get("tags", [])
        if not isinstance(tags, list):
            tags = []

        if not content:
            return json.dumps({"status": "failed", "error": "content is required"}, ensure_ascii=True)
        try:
            file_path = self._append_daily_memory(content=content, role=role, importance=importance, tags=tags)
            return json.dumps(
                {
                    "status": "ok",
                    "file": file_path.relative_to(self.workspace_root).as_posix(),
                    "role": role,
                    "importance": max(1, min(int(importance), 10)),
                },
                ensure_ascii=True,
            )
        except Exception as e:
            return json.dumps({"status": "failed", "error": str(e)}, ensure_ascii=True)

    async def memory_promote(self, kwargs_dict):
        kwargs = kwargs_dict or {}
        fact = str(kwargs.get("fact", "")).strip()
        if self.redact_sensitive_text:
            fact = self.redact_sensitive_text(fact)
        importance = int(kwargs.get("importance", 7))
        tags = kwargs.get("tags", [])
        if not isinstance(tags, list):
            tags = []
        if not fact:
            return json.dumps({"status": "failed", "error": "fact is required"}, ensure_ascii=True)

        try:
            self._ensure_memory_paths()
            now = datetime.now().isoformat(timespec="seconds")
            safe_importance = max(1, min(int(importance), 10))
            tag_text = ",".join(sorted({str(tag).strip() for tag in tags if str(tag).strip()}))
            entry = f"- [{now}] (importance:{safe_importance}) {fact}"
            if tag_text:
                entry += f" [tags:{tag_text}]"
            entry += "\n"
            with self.long_term_memory_file.open("a", encoding="utf-8", errors="replace", newline="") as fh:
                fh.write(entry)
            return json.dumps(
                {
                    "status": "ok",
                    "file": self.long_term_memory_file.relative_to(self.workspace_root).as_posix(),
                    "importance": safe_importance,
                },
                ensure_ascii=True,
            )
        except Exception as e:
            return json.dumps({"status": "failed", "error": str(e)}, ensure_ascii=True)

    async def memory_get(self, kwargs_dict=None):
        kwargs = kwargs_dict or {}
        date_str = str(kwargs.get("date", "")).strip()
        file_value = str(kwargs.get("file", "")).strip()
        start_line = int(kwargs.get("start_line", 1))
        end_line = int(kwargs.get("end_line", 200))
        max_chars = int(kwargs.get("max_chars", 20000))
        max_chars = max(200, min(max_chars, 500000))

        try:
            if start_line < 1:
                start_line = 1
            if end_line < start_line:
                end_line = start_line
            if file_value:
                path_obj = self.resolve_workspace_path(file_value, must_exist=True)
            elif date_str:
                target_date = datetime.strptime(date_str, "%Y-%m-%d")
                path_obj = self._daily_memory_file(target_date)
                if not path_obj.exists():
                    return json.dumps({"status": "failed", "error": f"No daily memory for {date_str}"}, ensure_ascii=True)
            else:
                path_obj = self.long_term_memory_file
                if not path_obj.exists():
                    self._ensure_memory_paths()
            text, lines = self._read_file_lines(path_obj)
            total_lines = len(lines)
            safe_end = min(end_line, total_lines)
            snippet = "\n".join(lines[start_line - 1:safe_end])
            snippet = snippet[:max_chars]
            return json.dumps(
                {
                    "status": "ok",
                    "file": path_obj.relative_to(self.workspace_root).as_posix(),
                    "start_line": start_line,
                    "end_line": safe_end,
                    "total_lines": total_lines,
                    "content": snippet,
                    "truncated": len(snippet) >= max_chars,
                },
                ensure_ascii=True,
            )
        except Exception as e:
            return json.dumps({"status": "failed", "error": str(e)}, ensure_ascii=True)

    async def memory_search(self, kwargs_dict):
        kwargs = kwargs_dict or {}
        query = str(kwargs.get("query", "")).strip()
        top_k = int(kwargs.get("top_k", 6))
        include_long_term = bool(kwargs.get("include_long_term", True))
        days_back = int(kwargs.get("days_back", 14))
        use_semantic = bool(kwargs.get("use_semantic", True))
        rebuild_index = bool(kwargs.get("rebuild_index", False))
        semantic_weight = float(kwargs.get("semantic_weight", 0.65))
        lexical_weight = float(kwargs.get("lexical_weight", 0.35))
        max_chunk_chars = int(kwargs.get("max_chunk_chars", 1600))
        top_k = max(1, min(top_k, 30))
        days_back = max(1, min(days_back, 120))
        max_chunk_chars = max(400, min(max_chunk_chars, 5000))
        semantic_weight = max(0.0, min(semantic_weight, 1.0))
        lexical_weight = max(0.0, min(lexical_weight, 1.0))
        if semantic_weight == 0.0 and lexical_weight == 0.0:
            lexical_weight = 1.0
        weight_sum = semantic_weight + lexical_weight
        semantic_weight = semantic_weight / weight_sum
        lexical_weight = lexical_weight / weight_sum

        if not query:
            return json.dumps({"status": "failed", "error": "query is required"}, ensure_ascii=True)

        try:
            self._ensure_memory_paths()
            candidates = []
            today = datetime.now()
            for offset in range(days_back):
                day = today - timedelta(days=offset)
                file_path = self._daily_memory_file(day)
                if file_path.exists():
                    candidates.append(file_path)
            if include_long_term and self.long_term_memory_file.exists():
                candidates.append(self.long_term_memory_file)

            if not candidates:
                return json.dumps(
                    {
                        "status": "ok",
                        "query": query,
                        "count": 0,
                        "results": [],
                        "semantic_enabled": use_semantic,
                    },
                    ensure_ascii=True,
                )

            query_tokens = self._tokenize_for_memory(query)
            lexical_by_id = {}
            for path_obj in candidates:
                raw = path_obj.read_text(encoding="utf-8", errors="replace")
                chunks = self._memory_chunks_from_text(raw, max_chunk_chars=max_chunk_chars)
                recency = self._memory_file_recency_boost(path_obj)
                file_rel = path_obj.relative_to(self.workspace_root).as_posix()
                for idx, chunk in enumerate(chunks, start=1):
                    chunk_hash = hashlib.sha1(chunk.encode("utf-8", errors="ignore")).hexdigest()
                    chunk_id = f"{file_rel}|{idx}|{chunk_hash[:16]}"
                    lexical = self._memory_chunk_score(query_tokens, chunk, recency_boost=recency)
                    lexical_by_id[chunk_id] = {
                        "file": file_rel,
                        "chunk_index": idx,
                        "chunk_hash": chunk_hash,
                        "snippet": chunk[:1200],
                        "lexical_score_raw": float(max(0.0, lexical)),
                        "recency_boost": float(recency),
                    }

            semantic_by_id = {}
            index_backend = "none"
            index_items = 0
            if use_semantic:
                index_data = self._build_memory_vector_index(
                    candidates=candidates,
                    force_rebuild=rebuild_index,
                    max_chunk_chars=max_chunk_chars,
                )
                index_backend = str(index_data.get("backend", "hash-embedding-v1"))
                index_items = len(index_data.get("items", []))
                query_vector = self._hash_embed_text(query, dim=int(index_data.get("dimension", self.vector_dim)))
                for item in index_data.get("items", []):
                    if not isinstance(item, dict):
                        continue
                    item_id = str(item.get("id", "")).strip()
                    if not item_id:
                        continue
                    semantic = self._cosine_similarity(query_vector, item.get("vector", []))
                    semantic = max(0.0, float(semantic))
                    semantic_by_id[item_id] = {
                        "semantic_score_raw": semantic,
                        "file": item.get("file", ""),
                        "chunk_index": int(item.get("chunk_index", 0) or 0),
                        "snippet": str(item.get("snippet", ""))[:1200],
                        "recency_boost": float(item.get("recency_boost", 0.0)),
                    }

            max_lexical = max((row.get("lexical_score_raw", 0.0) for row in lexical_by_id.values()), default=0.0)
            all_ids = set(lexical_by_id.keys()) | set(semantic_by_id.keys())
            ranked = []
            for chunk_id in all_ids:
                lexical_row = lexical_by_id.get(chunk_id, {})
                semantic_row = semantic_by_id.get(chunk_id, {})
                lexical_raw = float(lexical_row.get("lexical_score_raw", 0.0))
                lexical_norm = lexical_raw / max_lexical if max_lexical > 0 else 0.0
                semantic_raw = float(semantic_row.get("semantic_score_raw", 0.0))
                recency = float(lexical_row.get("recency_boost", semantic_row.get("recency_boost", 0.0)))
                combined = (lexical_weight * lexical_norm) + (semantic_weight * semantic_raw) + (0.03 * recency)
                if combined <= 0:
                    continue
                file_rel = lexical_row.get("file") or semantic_row.get("file") or ""
                chunk_index = int(lexical_row.get("chunk_index") or semantic_row.get("chunk_index") or 0)
                snippet = lexical_row.get("snippet") or semantic_row.get("snippet") or ""
                ranked.append(
                    {
                        "score": combined,
                        "combined_score": combined,
                        "lexical_score": lexical_norm,
                        "semantic_score": semantic_raw,
                        "file": file_rel,
                        "chunk_index": chunk_index,
                        "snippet": snippet[:1200],
                    }
                )

            ranked.sort(key=lambda item: item["score"], reverse=True)
            results = ranked[:top_k]
            return json.dumps(
                {
                    "status": "ok",
                    "query": query,
                    "count": len(results),
                    "semantic_enabled": use_semantic,
                    "weights": {"lexical": lexical_weight, "semantic": semantic_weight},
                    "vector_index": {
                        "backend": index_backend,
                        "file": self.vector_index_file.relative_to(self.workspace_root).as_posix(),
                        "items": index_items,
                    },
                    "results": results,
                },
                ensure_ascii=True,
            )
        except Exception as e:
            return json.dumps({"status": "failed", "error": str(e)}, ensure_ascii=True)

    async def memory_bootstrap(self, kwargs_dict=None):
        kwargs = kwargs_dict or {}
        include_long_term = bool(kwargs.get("include_long_term", True))
        max_chars = int(kwargs.get("max_chars", 24000))
        max_chars = max(1000, min(max_chars, 200000))

        try:
            self._ensure_memory_paths()
            today = datetime.now()
            yesterday = today - timedelta(days=1)
            files = []
            today_file = self._daily_memory_file(today)
            yday_file = self._daily_memory_file(yesterday)
            if today_file.exists():
                files.append(today_file)
            if yday_file.exists():
                files.append(yday_file)
            if include_long_term and self.long_term_memory_file.exists():
                files.append(self.long_term_memory_file)

            parts = []
            for file_path in files:
                text = file_path.read_text(encoding="utf-8", errors="replace")
                rel = file_path.relative_to(self.workspace_root).as_posix()
                parts.append(f"### {rel}\n{text.strip()}\n")

            merged = "\n\n".join(parts).strip()[:max_chars]
            return json.dumps(
                {
                    "status": "ok",
                    "files": [file_path.relative_to(self.workspace_root).as_posix() for file_path in files],
                    "content": merged,
                    "truncated": len(merged) >= max_chars,
                },
                ensure_ascii=True,
            )
        except Exception as e:
            return json.dumps({"status": "failed", "error": str(e)}, ensure_ascii=True)

    async def memory_reindex(self, kwargs_dict=None):
        kwargs = kwargs_dict or {}
        include_long_term = bool(kwargs.get("include_long_term", True))
        days_back = int(kwargs.get("days_back", 30))
        max_chunk_chars = int(kwargs.get("max_chunk_chars", 1600))
        force_rebuild = bool(kwargs.get("force_rebuild", True))
        days_back = max(1, min(days_back, 365))
        max_chunk_chars = max(400, min(max_chunk_chars, 5000))

        try:
            self._ensure_memory_paths()
            candidates = []
            today = datetime.now()
            for offset in range(days_back):
                day = today - timedelta(days=offset)
                file_path = self._daily_memory_file(day)
                if file_path.exists():
                    candidates.append(file_path)
            if include_long_term and self.long_term_memory_file.exists():
                candidates.append(self.long_term_memory_file)

            index_data = self._build_memory_vector_index(
                candidates=candidates,
                force_rebuild=force_rebuild,
                max_chunk_chars=max_chunk_chars,
            )
            return json.dumps(
                {
                    "status": "ok",
                    "files_indexed": len(candidates),
                    "vector_index_file": self.vector_index_file.relative_to(self.workspace_root).as_posix(),
                    "vector_backend": index_data.get("backend", "hash-embedding-v1"),
                    "dimension": int(index_data.get("dimension", self.vector_dim)),
                    "items": len(index_data.get("items", [])),
                    "updated_at": index_data.get("updated_at", ""),
                },
                ensure_ascii=True,
            )
        except Exception as e:
            return json.dumps({"status": "failed", "error": str(e)}, ensure_ascii=True)
