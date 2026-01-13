# SPEC: cc-search

**Version:** 1.0
**Status:** Approved
**Date:** 2026-01-13

---

## 1. PURPOSE

Solve lost knowledge and lack of continuity across Claude Code sessions. Enable both humans and Claude to search past conversations to recall solutions, understand project history, and build on previous work.

**Success:** Instant recall of relevant past context. Sessions become a searchable personal knowledge base. Claude can access its own history to build on previous work.

---

## 2. SCOPE

### IN SCOPE

- Semantic + keyword search across Claude Code sessions (~/.claude/projects)
- CLI for humans with human-readable output
- CLI for Claude (subprocess, parses stdout)
- Filtering by project, time range, content type
- Batch indexing (manual trigger)
- Local embeddings (sentence-transformers)

### OUT OF SCOPE (v1)

- Session management (delete, archive, cleanup)
- Cross-machine sync
- Real-time / automatic indexing
- TUI interface (backlog)
- Analytics / usage stats
- MCP server (CLI subprocess is sufficient)

---

## 3. DEFINITIONS

| Term | Definition |
|------|------------|
| Session | A single JSONL file in ~/.claude/projects containing a conversation |
| Message | One user or assistant turn within a session |
| Chunk | Searchable unit: user+assistant pair, sub-chunked if exceeds model max tokens |
| Content block | Component of assistant message (text, thinking, tool_use, tool_result) |
| Index | SQLite database with parsed sessions, FTS5 index, and sqlite-vec embeddings |

---

## 4. TECH STACK

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Language | Python 3.12+ | Best embeddings ecosystem |
| Packaging | uv | Clean distribution via uvx |
| Build backend | uv_build | Modern uv default, fast |
| Type checking | ty | Astral's fast type checker, integrates with uv |
| Linting | ruff | Fast, comprehensive |
| Storage | SQLite + FTS5 + sqlite-vec | Portable, no server, vectors in DB |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) | Local, free, ~100MB |
| CLI | typer | Subcommand support |
| Indexing | Batched encoding | Fast without multiprocessing complexity |

---

## 5. CLI INTERFACE

```
cc-search <command> [options]

Commands:
  search <query>    Search sessions
  chunk <id>        View full content of a chunk
  index             Build/rebuild the search index
  status            Show index stats

Search options:
  --project, -p     Filter to project (path substring match)
  --since, -s       Time range (e.g., "1w", "30d", "2024-01-01")
  --type, -t        Content type filter (user, assistant, tool, thinking)
  --limit, -n       Number of results (default: 5)
  --json            JSON output for programmatic use

Index options:
  --force           Reindex all sessions

Examples:
  cc-search search "authentication JWT"
  cc-search search "nginx config" --project myapp --since 2w
  cc-search search "caching strategy" --json
  cc-search chunk a2ffd9f7          # View full chunk from search results
  cc-search index
  cc-search status
```

---

## 6. OUTPUT FORMAT

### Human-readable (default)

```
[1] Project: myapp | 3 days ago | 100%

  You: How should I handle JWT refresh tokens?

  Claude: For JWT refresh tokens, you'll want to...
  [truncated - 847 more chars]

  → cc-search chunk a2ffd9f7

[2] Project: auth-service | 2 weeks ago | 87%
  ...

─────────────────────────────────────────────────
Found 2 results in 0.23s
```

**Color scheme:**
- `You:` label in cyan
- `Claude:` label in green
- Project name in green
- Rank number in bold cyan
- Age and score in dim

### JSON (--json flag)

```json
{
  "results": [
    {
      "rank": 1,
      "score": 0.87,
      "project": "myapp",
      "session_id": "abc123",
      "session_path": "...",
      "timestamp": "2024-01-10T14:32:00Z",
      "messages": [
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."}
      ]
    }
  ],
  "query": "authentication JWT",
  "total_results": 2,
  "search_time_ms": 234
}
```

---

## 7. CHUNKING STRATEGY

1. **Primary unit:** User + assistant message pairs
2. **Long message handling:** If pair exceeds model's max token limit (~512 for MiniLM), sub-chunk with overlap
3. **Sub-chunk metadata:** Maintains reference to parent session, message IDs, position
4. **Display:** Show matched chunk + reconstruct 3-5 message window for context

---

## 8. CONSTRAINTS

### PRIORITY ORDER

1. **Correctness** — Results are relevant and traceable
2. **Usability** — Useful for humans and Claude
3. **Performance** — Under 5 seconds
4. **Completeness** — All sessions indexed

### POSITIVE CONSTRAINTS (MUST)

| ID | Constraint | Verification |
|----|------------|--------------|
| P1 | Results in < 5 seconds | Benchmark |
| P2 | Filter by project, time, content type | Unit tests |
| P3 | Rank recent results higher (recency decay) | Manual |
| P4 | Semantic disambiguation from query context | Manual |
| P5 | 3-5 messages context per result | Inspection |
| P6 | Link to original session file | Unit test |
| P7 | Auto-index on first search if no index | Integration test |

### NEGATIVE CONSTRAINTS (MUST NOT)

| ID | Constraint | Rationale |
|----|------------|-----------|
| N1 | MUST NOT rank tool output above discussion by default | Noise reduction |
| N2 | MUST NOT rank thinking blocks above user-facing content | Internal vs external |
| N3 | MUST NOT show stale irrelevant above recent relevant | Recency matters |
| N4 | MUST NOT fail entirely on corrupted sessions | Partial value |

---

## 9. INVARIANTS

| ID | Property | Verification |
|----|----------|--------------|
| INV-1 | After reindex, every valid session is searchable | Compare counts |
| INV-2 | Every result links to existing session file | Unit test |
| INV-3 | Same query + same index = deterministic results | Repeat test |

---

## 10. EDGE CASES

| Case | Behavior |
|------|----------|
| No index exists | Auto-index, then search |
| Empty query | Error: "Query required" |
| No results | Show closest semantic matches with note |
| Corrupted session | Skip, log warning, continue |
| Long message in output | Truncate (human), full (JSON) |
| Project filter no match | "No sessions found for project X" |

---

## 11. DATA MODEL

```python
@dataclass
class Session:
    id: str                    # UUID from filename
    path: Path                 # Full path to JSONL
    project: str               # Extracted project name
    created_at: datetime
    updated_at: datetime

@dataclass
class Chunk:
    id: str
    session_id: str
    message_ids: list[str]     # Parent message references
    text: str                  # Combined text for embedding
    content_types: set[str]
    timestamp: datetime
    # embedding stored in sqlite-vec

@dataclass
class SearchResult:
    chunk: Chunk
    score: float
    session: Session
    context_messages: list[Message]  # Expanded for display
```

---

## 12. STORAGE SCHEMA

```sql
-- Sessions table
CREATE TABLE sessions (
    id TEXT PRIMARY KEY,
    path TEXT NOT NULL,
    project TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

-- Chunks table
CREATE TABLE chunks (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id),
    message_ids TEXT NOT NULL,  -- JSON array
    text TEXT NOT NULL,
    content_types TEXT NOT NULL,  -- JSON array
    timestamp TEXT NOT NULL
);

-- FTS5 for keyword search
CREATE VIRTUAL TABLE chunks_fts USING fts5(
    text,
    content='chunks',
    content_rowid='rowid'
);

-- sqlite-vec for embeddings
CREATE VIRTUAL TABLE chunks_vec USING vec0(
    embedding float[384]  -- MiniLM dimension
);
```

---

## 13. VERIFICATION

### Manual Checklist

| ID | Test | Expected | Status |
|----|------|----------|--------|
| M1 | Search known topic | Finds right session | ⬜ |
| M2 | Ambiguous term | Recent/relevant ranks higher | ⬜ |
| M3 | Project filter | No cross-contamination | ⬜ |
| M4 | Index completeness | All valid sessions indexed | ⬜ |
| M5 | JSON output | Valid, parseable by jq | ⬜ |
| M6 | First run | Auto-indexes successfully | ⬜ |

### Unit Tests

| Scenario | Given | When | Then |
|----------|-------|------|------|
| Basic search | Indexed sessions | Search keyword | Returns matches |
| Project filter | Multi-project index | Filter by project | Only that project |
| Time filter | Various dates | --since 7d | Only recent |
| Corrupted file | Invalid JSONL | Index | Skips, continues |
| Empty index | No index | Search | Auto-indexes first |

### Property Tests (Hypothesis)

| ID | Property | Generator | Assertion |
|----|----------|-----------|-----------|
| PT1 | Round-trip: index → search finds it | Random valid session content | If indexed, query containing its text returns it |
| PT2 | Filter consistency | Random project/time filters | Results always match filter criteria |
| PT3 | Ranking monotonicity | Same query, varying recency | More recent always scores ≥ older (same relevance) |
| PT4 | Determinism | Same query repeated N times | All N results identical |
| PT5 | Chunk reconstruction | Random long messages | Sub-chunks reference valid parent, can reconstruct |

```python
from hypothesis import given, strategies as st

@given(st.text(min_size=10))
def test_indexed_content_is_searchable(content):
    """If we index content, searching for it should find it."""
    # ... implementation
```

---

## 14. PROJECT STRUCTURE

Created via `uv init --package cc-search`:

```
cc-search/
├── .python-version          # Python version pin
├── .venv/                   # Virtual environment (uv managed)
├── pyproject.toml           # Project config
├── uv.lock                  # Lockfile (committed to git)
├── README.md
├── src/
│   └── cc_search/
│       ├── __init__.py
│       ├── py.typed         # PEP 561 marker
│       ├── cli.py           # Typer CLI
│       ├── indexer.py       # Parse JSONL, generate embeddings
│       ├── searcher.py      # Query, rank, format results
│       ├── storage.py       # SQLite + sqlite-vec operations
│       ├── embeddings.py    # sentence-transformers wrapper
│       ├── chunker.py       # Session → chunks logic
│       └── models.py        # Dataclasses
└── tests/
    ├── __init__.py
    ├── conftest.py          # Pytest fixtures
    ├── unit/
    │   ├── __init__.py
    │   ├── test_chunker.py
    │   ├── test_storage.py
    │   └── test_searcher.py
    ├── integration/
    │   ├── __init__.py
    │   ├── test_cli.py
    │   └── test_index_search.py
    └── property/
        ├── __init__.py
        ├── test_roundtrip.py
        ├── test_filters.py
        ├── test_ranking.py
        └── test_determinism.py
```

---

## 15. DEPENDENCIES

```toml
[project]
name = "cc-search"
version = "0.1.0"
description = "Search Claude Code session history with semantic + keyword search"
readme = "README.md"
requires-python = ">=3.12"
dependencies = [
    "typer",
    "sentence-transformers",
    "sqlite-vec",
    "rich",
]

[project.scripts]
cc-search = "cc_search.cli:app"

[build-system]
requires = ["uv_build>=0.6.6,<0.7"]
build-backend = "uv_build"

[dependency-groups]
dev = [
    "pytest",
    "hypothesis",
    "ruff",
]

[tool.ty]
python-version = "3.12"

[tool.ty.rules]
# Start permissive, tighten as codebase matures
possibly-unresolved-reference = "warn"

[tool.ruff]
line-length = 100
target-version = "py312"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "SIM"]
```

**Note:** No version pins on dependencies. Let `uv lock` resolve to latest compatible versions.

---

## 16. INDEX LOCATION

`~/.local/share/cc-search/index.db`

---

## 17. OPEN QUESTIONS (Resolved)

All open questions have been resolved:

- ✅ **Chunking strategy:** User+assistant pairs, sub-chunk if exceeds model max
- ✅ **Embedding scope:** Chunks (pairs), not individual messages
- ✅ **Vector storage:** sqlite-vec extension
- ✅ **Language:** Python with uv
- ✅ **Embeddings:** Local (sentence-transformers)
