"""Search functionality combining FTS5 and vector similarity."""

import re
from datetime import datetime, timedelta
from typing import Any

import sqlite_vec
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from cc_search.chunker import get_content_type_weight
from cc_search.embeddings import encode_text
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
    """Parse a since string into a datetime.

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

        now = datetime.now()
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
            return datetime.fromisoformat(since)
        else:
            return datetime.fromisoformat(since + "T00:00:00")
    except ValueError:
        console.print(f"[red]Invalid date format: {since}[/red]")
        return None


def calculate_recency_score(timestamp: datetime) -> float:
    """Calculate a recency score (0-1) with decay.

    More recent = higher score.
    Uses exponential decay with half-life of 30 days.
    """
    now = datetime.now()
    if timestamp.tzinfo is not None:
        timestamp = timestamp.replace(tzinfo=None)

    age_days = (now - timestamp).total_seconds() / 86400
    half_life = 30  # days

    # Exponential decay
    return 0.5 ** (age_days / half_life)


def search_fts(
    conn,
    query: str,
    project: str | None = None,
    since: datetime | None = None,
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

    if since:
        sql += " AND c.timestamp >= ?"
        params.append(since.isoformat())

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
    since: datetime | None = None,
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

        if since and chunk.timestamp < since:
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
    _query: str,
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

    for i, result in enumerate(results, 1):
        # Calculate relative time
        age = datetime.now() - result.session.updated_at.replace(tzinfo=None)
        if age.days > 0:
            age_str = f"{age.days} days ago"
        elif age.seconds > 3600:
            age_str = f"{age.seconds // 3600} hours ago"
        else:
            age_str = f"{age.seconds // 60} minutes ago"

        # Truncate text for display
        text = result.chunk.text
        if len(text) > 500:
            text = text[:500] + f"\n[truncated - {len(result.chunk.text) - 500} more chars]"

        # Create panel
        header = Text()
        header.append(f"[{i}] ", style="bold cyan")
        header.append(f"Project: {result.session.project}", style="green")
        header.append(f" | {age_str}", style="dim")
        header.append(f" | session {result.session.id[:8]}", style="dim")

        panel = Panel(
            text,
            title=str(header),
            subtitle=f"→ Full session: {result.session.path}",
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


def perform_search(
    query: str,
    project: str | None = None,
    since: str | None = None,
    content_type: str | None = None,
    limit: int = 5,
    json_output: bool = False,
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

    # Generate query embedding
    query_embedding = encode_text(query)

    # Perform both searches
    search_limit = limit * 5
    fts_results = search_fts(conn, query, project, since_dt, content_type, limit=search_limit)
    vector_results = search_vector(
        conn, query_embedding, project, since_dt, content_type, limit=search_limit
    )

    # Combine and rank
    ranked = combine_scores(fts_results, vector_results, conn)

    # Build results
    results = build_search_results(conn, ranked, limit)

    conn.close()

    search_time_ms = int((time.time() - start_time) * 1000)

    # Output
    if json_output:
        format_json_output(results, query, search_time_ms)
    else:
        format_human_output(results, query, search_time_ms, project_filter=project)
