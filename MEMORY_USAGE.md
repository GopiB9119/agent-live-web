# Memory Usage Guide

This guide shows exact tool-call payloads for every memory feature implemented in the agent.

## Storage Layout
- Daily logs: `memory/YYYY-MM-DD.md`
- Curated long-term memory: `MEMORY.md`
- Semantic vector index: `memory/.vector_index.json`

---

## 1) `memory_log`
Append a memory event to today's daily file.

Tool call:
```json
{
  "tool_name": "memory_log",
  "arguments": {
    "content": "User prefers Edge browser automation over Chromium.",
    "role": "user",
    "importance": 6,
    "tags": ["preference", "browser"]
  }
}
```

Expected result shape:
```json
{
  "status": "ok",
  "file": "memory/2026-03-01.md",
  "role": "user",
  "importance": 6
}
```

---

## 2) `memory_bootstrap`
Load startup memory context (today, yesterday, and optionally long-term memory).

Tool call:
```json
{
  "tool_name": "memory_bootstrap",
  "arguments": {
    "include_long_term": true,
    "max_chars": 24000
  }
}
```

Expected result shape:
```json
{
  "status": "ok",
  "files": ["memory/2026-03-01.md", "memory/2026-02-28.md", "MEMORY.md"],
  "content": "### memory/2026-03-01.md ...",
  "truncated": false
}
```

---

## 3) `memory_get`
Read a targeted memory range by date or file.

Read a daily file by date:
```json
{
  "tool_name": "memory_get",
  "arguments": {
    "date": "2026-03-01",
    "start_line": 1,
    "end_line": 120,
    "max_chars": 20000
  }
}
```

Read long-term memory by file:
```json
{
  "tool_name": "memory_get",
  "arguments": {
    "file": "MEMORY.md",
    "start_line": 1,
    "end_line": 200
  }
}
```

Expected result shape:
```json
{
  "status": "ok",
  "file": "MEMORY.md",
  "start_line": 1,
  "end_line": 80,
  "total_lines": 80,
  "content": "...",
  "truncated": false
}
```

---

## 4) `memory_search` (Hybrid Lexical + Semantic)
Search memory with combined lexical score and embedding-style vector score.

Tool call:
```json
{
  "tool_name": "memory_search",
  "arguments": {
    "query": "edge browser profile issue fixed with single owned tab",
    "top_k": 6,
    "include_long_term": true,
    "days_back": 30,
    "use_semantic": true,
    "semantic_weight": 0.65,
    "lexical_weight": 0.35,
    "rebuild_index": false
  }
}
```

Expected result shape:
```json
{
  "status": "ok",
  "query": "edge browser profile issue fixed with single owned tab",
  "count": 6,
  "semantic_enabled": true,
  "weights": {
    "lexical": 0.35,
    "semantic": 0.65
  },
  "vector_index": {
    "backend": "hash-embedding-v1",
    "file": "memory/.vector_index.json",
    "items": 42
  },
  "results": [
    {
      "score": 0.91,
      "combined_score": 0.91,
      "lexical_score": 0.68,
      "semantic_score": 0.95,
      "file": "memory/2026-03-01.md",
      "chunk_index": 4,
      "snippet": "..."
    }
  ]
}
```

---

## 5) `memory_reindex`
Rebuild memory vector index after major memory updates.

Tool call:
```json
{
  "tool_name": "memory_reindex",
  "arguments": {
    "include_long_term": true,
    "days_back": 45,
    "max_chunk_chars": 1600,
    "force_rebuild": true
  }
}
```

Expected result shape:
```json
{
  "status": "ok",
  "files_indexed": 4,
  "vector_index_file": "memory/.vector_index.json",
  "vector_backend": "hash-embedding-v1",
  "dimension": 192,
  "items": 57,
  "updated_at": "2026-03-01T19:30:00"
}
```

---

## 6) `memory_promote`
Promote durable facts to long-term curated memory.

Tool call:
```json
{
  "tool_name": "memory_promote",
  "arguments": {
    "fact": "User wants local Microsoft Edge as the only browser for Playwright sessions.",
    "importance": 9,
    "tags": ["preference", "edge", "playwright"]
  }
}
```

Expected result shape:
```json
{
  "status": "ok",
  "file": "MEMORY.md",
  "importance": 9
}
```

---

## Recommended Sequence
1. `memory_bootstrap` at session start.
2. `memory_log` for important user and assistant events.
3. `memory_search` for recall during planning/execution.
4. `memory_get` for precise line-range evidence.
5. `memory_promote` for long-lived facts/preferences.
6. `memory_reindex` after many writes or promotions.

