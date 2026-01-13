# Implementation Prompt: cc-search

You are implementing `cc-search`, a CLI tool to search Claude Code session history using semantic + keyword search. The full specification is in `SPEC.md` in this directory. Read it thoroughly before starting.

## Quick Summary

**What:** CLI tool to search `~/.claude/projects/**/*.jsonl` sessions
**Tech:** Python 3.12+, uv, SQLite + FTS5 + sqlite-vec, sentence-transformers, typer
**Distribution:** `uvx cc-search` or `uv tool install cc-search`

## Critical: uv Best Practices

### Project Setup

The project should be initialized as a **packaged application** (src layout):

```bash
cd ~/Code/cc-search
uv init --package .
```

This creates:
- `src/cc_search/` directory with `__init__.py`
- `pyproject.toml` with `[build-system]` using `uv_build`
- `.python-version` file
- `.venv/` directory (managed by uv)

### Dependency Management

**Add runtime dependencies:**
```bash
uv add typer sentence-transformers sqlite-vec rich
```

**Add dev dependencies (creates `[dependency-groups]` section):**
```bash
uv add --dev pytest hypothesis ruff
```

**After any dependency changes:**
```bash
uv sync
```

**Commit these files to git:**
- `pyproject.toml`
- `uv.lock`
- `.python-version`

### Running Commands

**Always use `uv run` to execute commands in the project environment:**
```bash
uv run cc-search search "query"     # Run the CLI
uv run pytest                        # Run tests
uv run python -c "..."               # Run Python
```

**Use `uvx` for one-off tool execution:**
```bash
uvx ty check                         # Type check with ty
uvx ruff check .                     # Lint with ruff
uvx ruff format .                    # Format with ruff
```

### Type Checking with ty

ty is Astral's fast type checker. Configure in `pyproject.toml`:

```toml
[tool.ty]
python-version = "3.12"

[tool.ty.rules]
possibly-unresolved-reference = "warn"
```

**Run type checking:**
```bash
uvx ty check
```

**Key ty features:**
- 10-100x faster than mypy/pyright
- Supports `type: ignore` comments
- Configurable rule severity: `ignore`, `warn`, `error`
- Per-file overrides via `[[tool.ty.overrides]]`

### Linting with ruff

Configure in `pyproject.toml`:

```toml
[tool.ruff]
line-length = 100
target-version = "py312"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "SIM"]
```

**Run linting:**
```bash
uvx ruff check .       # Check
uvx ruff check . --fix # Auto-fix
uvx ruff format .      # Format
```

### Building and Testing Distribution

**Build the package:**
```bash
uv build
```

**Test that uvx works:**
```bash
uvx --from . cc-search --help
```

**Test installation:**
```bash
uv tool install .
cc-search --help
uv tool uninstall cc-search
```

## pyproject.toml Reference

The full `pyproject.toml` should look like this (see SPEC.md Section 15 for details):

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
possibly-unresolved-reference = "warn"

[tool.ruff]
line-length = 100
target-version = "py312"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "SIM"]

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_functions = ["test_*"]
```

## Implementation Order

Follow this order to ensure you can verify each step:

### Phase 1: Project Scaffold

1. Initialize project structure with `uv init --package`
2. Configure `pyproject.toml` as specified
3. Add dependencies with `uv add`
4. Verify: `uv run cc-search --help` shows help

### Phase 2: Core Data Layer

1. Implement `models.py` - dataclasses for Session, Chunk, Message, SearchResult
2. Implement `storage.py` - SQLite schema creation, CRUD operations
3. Verify: Unit tests pass for storage operations

### Phase 3: JSONL Parsing

1. Implement `indexer.py` - parse `~/.claude/projects/**/*.jsonl`
2. Handle the JSONL format (see SPEC.md Section 3 for structure)
3. Extract user messages, assistant text (skip thinking blocks in default ranking)
4. Verify: Can parse your actual session files

### Phase 4: Chunking

1. Implement `chunker.py` - split sessions into user+assistant pairs
2. Handle long messages (sub-chunk if > model max tokens)
3. Verify: Chunks are correctly formed with metadata

### Phase 5: Embeddings

1. Implement `embeddings.py` - sentence-transformers wrapper
2. Use `all-MiniLM-L6-v2` model (384 dimensions)
3. Batch encode for efficiency
4. Verify: Can generate embeddings for sample text

### Phase 6: Search

1. Implement `searcher.py` - combine FTS5 + vector similarity
2. Implement recency decay in ranking
3. Implement filtering (project, time, content type)
4. Verify: Search returns relevant results

### Phase 7: CLI

1. Implement `cli.py` - typer app with subcommands
2. Commands: `search`, `index`, `status`
3. Human-readable and JSON output formats
4. Verify: Full CLI works end-to-end

### Phase 8: Testing

1. Unit tests for each module
2. Integration tests for CLI
3. Property tests with Hypothesis (see SPEC.md Section 13)
4. Verify: All tests pass, type check passes

## Verification Checklist

After implementation, verify these work:

```bash
# Type checking
uvx ty check

# Linting
uvx ruff check .

# Tests
uv run pytest

# CLI help
uv run cc-search --help

# Index sessions
uv run cc-search index

# Search
uv run cc-search search "authentication"
uv run cc-search search "auth" --project myproject --since 1w
uv run cc-search search "test" --json | jq .

# Status
uv run cc-search status

# Distribution test
uvx --from . cc-search --help
```

## Key Files in Claude Code Sessions

The JSONL files in `~/.claude/projects/` have this structure:

**Project folder naming:** `/Users/name/Code/project` → `-Users-name-Code-project`

**JSONL record types:**
- `type: "user"` - user messages
- `type: "assistant"` - Claude responses with content blocks
- `type: "file-history-snapshot"` - file state tracking

**Assistant message content blocks:**
- `text` - the actual response (INDEX THIS)
- `thinking` - internal reasoning (lower ranking weight)
- `tool_use` - tool invocations
- `tool_result` - tool outputs (lower ranking weight)

## Important Constraints

From SPEC.md Section 8:

- **N1:** MUST NOT rank tool output above discussion by default
- **N2:** MUST NOT rank thinking blocks above user-facing content
- **N3:** MUST NOT show stale irrelevant above recent relevant
- **N4:** MUST NOT fail entirely on corrupted sessions (skip and warn)

## Index Location

Store the SQLite database at: `~/.local/share/cc-search/index.db`

Create the directory if it doesn't exist.

## Reference

- Full specification: `./SPEC.md`
- uv documentation: https://docs.astral.sh/uv/
- ty documentation: https://docs.astral.sh/ty/
- sqlite-vec: https://github.com/asg017/sqlite-vec
- sentence-transformers: https://www.sbert.net/
