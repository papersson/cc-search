# cc-search

Semantic + keyword search for Claude Code sessions.

## Install

```bash
uvx cc-search --help
```

Or install globally:

```bash
uv tool install cc-search
```

## Usage

```bash
# Build the index (first time)
cc-search index

# Search your sessions
cc-search search "authentication JWT"
cc-search search "nginx config" --project myapp --since 2w

# View full content of a result
cc-search chunk a2ffd9f7

# Check index status
cc-search status
```

## Options

```
--project, -p    Filter by project (path substring)
--since, -s      Time range: 1d, 2w, 30d, 2024-01-01
--type, -t       Content type: user, assistant, tool, thinking
--limit, -n      Number of results (default: 5)
--json           JSON output
```

## How it works

- Indexes `~/.claude/projects/**/*.jsonl`
- Hybrid search: SQLite FTS5 + vector similarity (all-MiniLM-L6-v2)
- Results ranked by relevance + recency
