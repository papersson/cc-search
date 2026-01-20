# cc-search

Semantic + keyword search for Claude Code sessions.

## Install

```bash
uv tool install git+https://github.com/papersson/cc-search
```

Or clone and install locally:

```bash
git clone https://github.com/papersson/cc-search
cd cc-search
uv tool install .
```

Or run without installing:

```bash
uvx --from git+https://github.com/papersson/cc-search cc-search --help
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

# List all projects
cc-search projects
```

## Search Options

```
--project, -p      Filter by project (path substring)
--exclude, -x      Exclude projects (can repeat)
--since, -s        Start time: 1d, 2w, 30d, 2024-01-01
--until, -u        End time: 1d, 2024-06-30
--type, -t         Content type: user, assistant, tool, thinking
--limit, -n        Number of results (default: 5)
--json             JSON output
--paths            Only output session paths
--interactive, -i  Interactive selection with fzf
--export, -e       Export results to markdown file
```

## Chunk Options

```
--open, -o       Open session file in $EDITOR
--path           Only output session path
--context, -C    Number of surrounding chunks to show
```

## Index Options

```
--force, -f      Reindex all sessions
--dry-run, -d    Show what would be indexed
```

## Status Options

```
--verbose, -v    Show per-project breakdown
```

## How it works

- Indexes `~/.claude/projects/**/*.jsonl`
- Hybrid search: SQLite FTS5 + vector similarity (all-MiniLM-L6-v2)
- Results ranked by relevance + recency
- Query terms highlighted in results
