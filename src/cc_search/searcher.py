"""Search functionality combining FTS5 and vector similarity."""

import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import sqlite_vec
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from cc_search.chunker import get_content_type_weight
from cc_search.models import SearchResult
from cc_search.storage import (
    ensure_index_exists,
    get_chunk,
    get_session,
    index_exists,
)

console = Console()

# Weights for combining search scores
FTS_WEIGHT = 0.4
VECTOR_WEIGHT = 0.6
RECENCY_WEIGHT = 0.2


def parse_since(since: str | None) -> datetime | None:
    """Parse a since string into a UTC-aware datetime.

    Supports:
    - Relative: "1w", "7d", "30d", "2h"
    - Absolute: "2024-01-01", "2024-01-01T00:00:00"
    """
    if since is None:
        return None

    since = since.strip().lower()

    # Relative time patterns
    match = re.match(r"^(\d+)([hdwmy])$", since)
    if match:
        amount = int(match.group(1))
        unit = match.group(2)

        now = datetime.now(tz=timezone.utc)
        if unit == "h":
            return now - timedelta(hours=amount)
        elif unit == "d":
            return now - timedelta(days=amount)
        elif unit == "w":
            return now - timedelta(weeks=amount)
        elif unit == "m":
            return now - timedelta(days=amount * 30)  # Approximate
        elif unit == "y":
            return now - timedelta(days=amount * 365)  # Approximate

    # Try parsing as ISO date
    try:
        if "T" in since:
            dt = datetime.fromisoformat(since)
        else:
            dt = datetime.fromisoformat(since + "T00:00:00")
        # If naive, assume UTC
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        console.print(f"[red]Invalid date format: {since}[/red]")
        return None


def calculate_recency_score(timestamp: datetime) -> float:
    """Calculate a recency score (0-1) with decay.

    More recent = higher score.
    Uses exponential decay with half-life of 30 days.
    """
    now = datetime.now(tz=timezone.utc)
    # Ensure timestamp is tz-aware for comparison
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)

    age_days = (now - timestamp).total_seconds() / 86400
    half_life = 30  # days

    # Exponential decay
    return 0.5 ** (age_days / half_life)


def search_fts(
    conn,
    query: str,
    project: str | None = None,
    exclude: list[str] | None = None,
    since: datetime | None = None,
    until: datetime | None = None,
    content_type: str | None = None,
    limit: int = 50,
) -> list[tuple[str, float]]:
    """Search using FTS5 full-text search.

    Returns list of (chunk_id, bm25_score) tuples.
    """
    # Build the query with filters
    sql = """
        SELECT c.id, bm25(chunks_fts) as score
        FROM chunks_fts
        JOIN chunks c ON chunks_fts.rowid = c.rowid
        JOIN sessions s ON c.session_id = s.id
        WHERE chunks_fts MATCH ?
    """
    params: list[Any] = [query]

    if project:
        sql += " AND s.project LIKE ?"
        params.append(f"%{project}%")

    if exclude:
        for pattern in exclude:
            sql += " AND s.project NOT LIKE ?"
            params.append(f"%{pattern}%")

    if since:
        sql += " AND c.timestamp >= ?"
        params.append(since.isoformat())

    if until:
        sql += " AND c.timestamp <= ?"
        params.append(until.isoformat())

    if content_type:
        sql += " AND c.content_types LIKE ?"
        params.append(f'%"{content_type}"%')

    sql += " ORDER BY score LIMIT ?"
    params.append(limit)

    try:
        results = conn.execute(sql, params).fetchall()
        # BM25 scores are negative (lower is better), so negate them
        return [(row[0], -row[1]) for row in results]
    except Exception:
        # Query might fail if no FTS matches
        return []


def search_vector(
    conn,
    query_embedding: list[float],
    project: str | None = None,
    exclude: list[str] | None = None,
    since: datetime | None = None,
    until: datetime | None = None,
    content_type: str | None = None,
    limit: int = 50,
) -> list[tuple[str, float]]:
    """Search using vector similarity.

    Returns list of (chunk_id, similarity_score) tuples.
    """
    # Vector search with sqlite-vec
    serialized = sqlite_vec.serialize_float32(query_embedding)

    # First get vector matches
    sql = """
        SELECT chunk_id, distance
        FROM chunks_vec
        WHERE embedding MATCH ?
        ORDER BY distance
        LIMIT ?
    """

    try:
        vector_results = conn.execute(sql, (serialized, limit * 2)).fetchall()
    except Exception:
        return []

    # Filter by other criteria
    results: list[tuple[str, float]] = []
    for chunk_id, distance in vector_results:
        # Get chunk and session for filtering
        chunk = get_chunk(conn, chunk_id)
        if chunk is None:
            continue

        session = get_session(conn, chunk.session_id)
        if session is None:
            continue

        # Apply filters
        if project and project.lower() not in session.project.lower():
            continue

        if exclude:
            excluded = False
            for pattern in exclude:
                if pattern.lower() in session.project.lower():
                    excluded = True
                    break
            if excluded:
                continue

        if since and chunk.timestamp < since:
            continue

        if until and chunk.timestamp > until:
            continue

        if content_type and content_type not in chunk.content_types:
            continue

        # Convert distance to similarity (lower distance = higher similarity)
        similarity = 1.0 / (1.0 + distance)
        results.append((chunk_id, similarity))

        if len(results) >= limit:
            break

    return results


def combine_scores(
    fts_results: list[tuple[str, float]],
    vector_results: list[tuple[str, float]],
    conn,
) -> list[tuple[str, float]]:
    """Combine FTS and vector scores with recency weighting."""
    # Normalize scores
    fts_scores = dict(fts_results)
    vector_scores = dict(vector_results)

    # Normalize FTS scores (0-1)
    if fts_scores:
        max_fts = max(fts_scores.values())
        if max_fts > 0:
            fts_scores = {k: v / max_fts for k, v in fts_scores.items()}

    # Vector scores should already be 0-1 from similarity

    # Combine all chunk IDs
    all_chunk_ids = set(fts_scores.keys()) | set(vector_scores.keys())

    combined: list[tuple[str, float]] = []
    for chunk_id in all_chunk_ids:
        fts_score = fts_scores.get(chunk_id, 0)
        vec_score = vector_scores.get(chunk_id, 0)

        # Get chunk for recency and content type weighting
        chunk = get_chunk(conn, chunk_id)
        if chunk is None:
            continue

        # Calculate recency score
        recency = calculate_recency_score(chunk.timestamp)

        # Calculate content type weight (average of types in chunk)
        type_weights = [get_content_type_weight(t) for t in chunk.content_types]
        content_weight = sum(type_weights) / len(type_weights) if type_weights else 1.0

        # Combine scores
        base_score = (fts_score * FTS_WEIGHT) + (vec_score * VECTOR_WEIGHT)
        final_score = base_score * content_weight * (1 + recency * RECENCY_WEIGHT)

        combined.append((chunk_id, final_score))

    # Sort by score descending
    combined.sort(key=lambda x: x[1], reverse=True)
    return combined


def build_search_results(
    conn, ranked_chunks: list[tuple[str, float]], limit: int
) -> list[SearchResult]:
    """Build SearchResult objects from ranked chunks."""
    results: list[SearchResult] = []

    for chunk_id, score in ranked_chunks[:limit]:
        chunk = get_chunk(conn, chunk_id)
        if chunk is None:
            continue

        session = get_session(conn, chunk.session_id)
        if session is None:
            continue

        results.append(
            SearchResult(
                chunk=chunk,
                score=score,
                session=session,
                context_messages=[],  # Could expand to load surrounding messages
            )
        )

    return results


def format_human_output(
    results: list[SearchResult],
    query: str,
    search_time_ms: int,
    project_filter: str | None = None,
) -> None:
    """Format results for human-readable output."""
    if not results:
        if project_filter:
            console.print(f"[yellow]No sessions found for project '{project_filter}'[/yellow]")
        else:
            console.print("[yellow]No results found. Try a different query.[/yellow]")
        return

    # Calculate max score for percentage normalization
    max_score = max(r.score for r in results) if results else 1.0

    for i, result in enumerate(results, 1):
        # Calculate relative time
        updated_at = result.session.updated_at
        if updated_at.tzinfo is None:
            updated_at = updated_at.replace(tzinfo=timezone.utc)
        age = datetime.now(tz=timezone.utc) - updated_at
        if age.days > 0:
            age_str = f"{age.days} days ago"
        elif age.seconds > 3600:
            age_str = f"{age.seconds // 3600} hours ago"
        else:
            age_str = f"{age.seconds // 60} minutes ago"

        # Calculate score percentage (relative to top result)
        score_pct = int((result.score / max_score) * 100) if max_score > 0 else 0

        # Truncate text for display
        text = result.chunk.text
        if len(text) > 500:
            remaining = len(result.chunk.text) - 500
            text = text[:500] + f"\n[dim][truncated - {remaining} more chars][/dim]"

        # Apply highlighting and colorization
        text = highlight_matches(text, query)
        text = colorize_roles(text)

        # Create panel header
        header = Text()
        header.append(f"[{i}] ", style="bold cyan")
        header.append(f"Project: {result.session.project}", style="green")
        header.append(f" | {age_str}", style="dim")
        header.append(f" | {score_pct}%", style="dim")

        panel = Panel(
            text,
            title=str(header),
            subtitle=f"→ cc-search chunk {result.chunk.id[:8]}",
            subtitle_align="left",
        )
        console.print(panel)
        console.print()

    console.print("─" * 50)
    console.print(f"Found {len(results)} results in {search_time_ms}ms")


def parse_chunk_messages(text: str) -> list[dict[str, str]]:
    """Parse chunk text into user/assistant messages."""
    messages = []
    parts = re.split(r"\n\n(?=You:|Claude:)", text)
    for part in parts:
        part = part.strip()
        if part.startswith("You:"):
            messages.append({"role": "user", "content": part[4:].strip()})
        elif part.startswith("Claude:"):
            messages.append({"role": "assistant", "content": part[7:].strip()})
        elif part:
            messages.append({"role": "assistant", "content": part})
    return messages


def format_json_output(results: list[SearchResult], query: str, search_time_ms: int) -> None:
    """Format results as JSON for programmatic use."""
    output = {
        "results": [
            {
                "rank": i + 1,
                "score": round(result.score, 4),
                "project": result.session.project,
                "session_id": result.session.id,
                "session_path": str(result.session.path),
                "timestamp": result.chunk.timestamp.isoformat(),
                "messages": parse_chunk_messages(result.chunk.text),
            }
            for i, result in enumerate(results)
        ],
        "query": query,
        "total_results": len(results),
        "search_time_ms": search_time_ms,
    }
    console.print_json(data=output)


def colorize_roles(text: str) -> str:
    """Add Rich markup to colorize You:/Claude: labels."""
    text = text.replace("You:", "[cyan]You:[/cyan]")
    text = text.replace("Claude:", "[green]Claude:[/green]")
    return text


def highlight_matches(text: str, query: str) -> str:
    """Highlight query terms in text using Rich markup."""
    # Split query into words for individual highlighting
    terms = query.lower().split()

    for term in terms:
        # Skip very short terms to avoid too many highlights
        if len(term) < 3:
            continue
        # Case-insensitive replacement with markup
        pattern = re.compile(re.escape(term), re.IGNORECASE)
        text = pattern.sub(lambda m: f"[bold yellow]{m.group()}[/bold yellow]", text)

    return text


def format_paths_output(results: list[SearchResult]) -> None:
    """Print only unique session paths."""
    paths = sorted(set(str(r.session.path) for r in results))
    for path in paths:
        console.print(path)


def interactive_search(results: list[SearchResult]) -> None:
    """Run fzf on results and display selected chunk."""
    import subprocess

    if not results:
        console.print("[yellow]No results to select from[/yellow]")
        return

    # Format results as fzf-friendly lines
    lines = []
    for result in results:
        # Format: chunk_id[TAB]project[TAB]truncated_text
        text_preview = result.chunk.text[:100].replace("\n", " ")
        line = f"{result.chunk.id[:8]}\t{result.session.project}\t{text_preview}"
        lines.append(line)
    fzf_input = "\n".join(lines)

    try:
        proc = subprocess.run(
            ["fzf", "--with-nth=2,3", "--delimiter=\t", "--preview-window=wrap"],
            input=fzf_input,
            capture_output=True,
            text=True,
            check=True,
        )
        selected = proc.stdout.strip()
        if selected:
            chunk_id = selected.split("\t")[0]
            display_chunk(chunk_id)
    except FileNotFoundError:
        console.print("[red]fzf not found. Install with: brew install fzf[/red]")
    except subprocess.CalledProcessError:
        pass  # User cancelled fzf selection


def export_to_markdown(results: list[SearchResult], query: str, output_path: "Path") -> None:
    """Export search results to markdown file."""
    lines = [
        f"# Search Results for: {query}",
        "",
        f"*Generated: {datetime.now(tz=timezone.utc).isoformat()}*",
        f"*Results: {len(results)}*",
        "",
    ]

    for i, result in enumerate(results, 1):
        lines.extend([
            f"## [{i}] {result.session.project}",
            "",
            f"- **Session**: `{result.session.path}`",
            f"- **Time**: {result.chunk.timestamp.isoformat()}",
            f"- **Score**: {result.score:.4f}",
            "",
            "```",
            result.chunk.text,
            "```",
            "",
        ])

    output_path.write_text("\n".join(lines))
    console.print(f"[green]Exported {len(results)} results to {output_path}[/green]")


def reconstruct_full_text(chunks: list) -> str:
    """Reconstruct full text from potentially overlapping sub-chunks."""
    if len(chunks) == 1:
        return chunks[0].text

    # Sort chunks: the one starting with "You:" goes first, then by text content
    def sort_key(c):
        if c.text.strip().startswith("You:"):
            return (0, "")
        return (1, c.text[:100])

    sorted_chunks = sorted(chunks, key=sort_key)

    # Merge overlapping chunks
    result = sorted_chunks[0].text
    for chunk in sorted_chunks[1:]:
        chunk_text = chunk.text
        # Find overlap by checking if end of result matches start of chunk
        best_overlap = 0
        # Check overlaps from large to small (up to 600 chars which is > 25% of 2000)
        for overlap_size in range(min(600, len(result), len(chunk_text)), 50, -1):
            if result.endswith(chunk_text[:overlap_size]):
                best_overlap = overlap_size
                break
        # Append non-overlapping part
        if best_overlap > 0:
            result += chunk_text[best_overlap:]
        else:
            # No overlap found - chunks might not be contiguous, just append
            result += "\n\n" + chunk_text

    return result


def open_session_in_editor(session_path: Path, search_text: str | None = None) -> None:
    """Open session file in editor, optionally at a specific line."""
    import os
    import subprocess

    editor = os.environ.get("EDITOR", "vim")

    # Find line number if search text provided
    line_num = 1
    if search_text:
        try:
            with open(session_path) as f:
                for i, line in enumerate(f, 1):
                    if search_text[:50] in line:  # Match first 50 chars
                        line_num = i
                        break
        except Exception:
            pass

    # Build editor command with line number
    if editor in ("vim", "nvim", "vi"):
        cmd = [editor, f"+{line_num}", str(session_path)]
    elif editor in ("code", "code-insiders"):
        cmd = [editor, "--goto", f"{session_path}:{line_num}"]
    elif editor == "subl":
        cmd = [editor, f"{session_path}:{line_num}"]
    else:
        cmd = [editor, str(session_path)]

    subprocess.run(cmd)


def display_chunk(
    chunk_id: str,
    open_editor: bool = False,
    path_only: bool = False,
    context: int = 0,
) -> None:
    """Display full chunk content by ID (or prefix)."""
    from cc_search.storage import (
        ensure_index_exists,
        get_chunk_by_prefix,
        get_chunks_by_message_ids,
        get_session,
        get_session_chunks_ordered,
    )

    conn = ensure_index_exists()
    chunk = get_chunk_by_prefix(conn, chunk_id)

    if chunk is None:
        console.print(f"[red]Chunk not found: {chunk_id}[/red]")
        conn.close()
        return

    session = get_session(conn, chunk.session_id)

    # Handle path-only output
    if path_only:
        if session:
            console.print(str(session.path))
        conn.close()
        return

    # Handle open in editor
    if open_editor and session:
        conn.close()
        open_session_in_editor(session.path, chunk.text[:100])
        return

    # Get context chunks if requested
    if context > 0 and session:
        all_session_chunks = get_session_chunks_ordered(conn, session.id)
        # Find the index of the current chunk
        chunk_idx = next(
            (i for i, c in enumerate(all_session_chunks) if c.id == chunk.id),
            None
        )
        if chunk_idx is not None:
            start_idx = max(0, chunk_idx - context)
            end_idx = min(len(all_session_chunks), chunk_idx + context + 1)
            context_chunks = all_session_chunks[start_idx:end_idx]
            # Build full text from context chunks
            full_text = "\n\n---\n\n".join(c.text for c in context_chunks)
        else:
            # Fallback to just the current chunk
            all_chunks = get_chunks_by_message_ids(conn, chunk.message_ids)
            full_text = reconstruct_full_text(all_chunks)
    else:
        # Get all chunks with same message_ids to reconstruct full content
        all_chunks = get_chunks_by_message_ids(conn, chunk.message_ids)
        full_text = reconstruct_full_text(all_chunks)

    conn.close()

    # Calculate relative time
    timestamp = chunk.timestamp
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    age = datetime.now(tz=timezone.utc) - timestamp
    if age.days > 0:
        age_str = f"{age.days} days ago"
    elif age.seconds > 3600:
        age_str = f"{age.seconds // 3600} hours ago"
    else:
        age_str = f"{age.seconds // 60} minutes ago"

    # Build header
    header = Text()
    if session:
        header.append(f"Project: {session.project}", style="green")
        header.append(f" | {age_str}", style="dim")

    # Colorize the text
    colored_text = colorize_roles(full_text)

    # Create panel with full content
    panel = Panel(
        colored_text,
        title=str(header) if session else None,
        subtitle=f"→ Session: {session.path}" if session else None,
        subtitle_align="left",
    )
    console.print(panel)


def perform_search(
    query: str,
    project: str | None = None,
    exclude: list[str] | None = None,
    since: str | None = None,
    until: str | None = None,
    content_type: str | None = None,
    limit: int = 5,
    json_output: bool = False,
    paths_only: bool = False,
    interactive: bool = False,
    export_path: Path | None = None,
) -> None:
    """Perform a search and display results."""
    import time

    # Auto-index if no index exists (P7)
    if not index_exists():
        console.print("[yellow]No index found, building index first...[/yellow]")
        from cc_search.indexer import build_index

        build_index()

    start_time = time.time()

    conn = ensure_index_exists()
    since_dt = parse_since(since)
    until_dt = parse_since(until)

    # Generate query embedding (lazy import to avoid loading torch for non-search commands)
    from cc_search.embeddings import encode_text

    query_embedding = encode_text(query)

    # Perform both searches
    search_limit = limit * 5
    fts_results = search_fts(
        conn, query, project, exclude, since_dt, until_dt, content_type, limit=search_limit
    )
    vector_results = search_vector(
        conn, query_embedding, project, exclude, since_dt, until_dt, content_type, limit=search_limit
    )

    # Combine and rank
    ranked = combine_scores(fts_results, vector_results, conn)

    # Build results
    results = build_search_results(conn, ranked, limit)

    conn.close()

    search_time_ms = int((time.time() - start_time) * 1000)

    # Output based on mode
    if interactive:
        interactive_search(results)
    elif paths_only:
        format_paths_output(results)
    elif export_path:
        export_to_markdown(results, query, export_path)
    elif json_output:
        format_json_output(results, query, search_time_ms)
    else:
        format_human_output(results, query, search_time_ms, project_filter=project)
