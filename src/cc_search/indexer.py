"""JSONL session indexer."""

import json
from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from cc_search.chunker import create_chunks
from cc_search.embeddings import encode_batch
from cc_search.models import Message, Session
from cc_search.storage import (
    delete_session_chunks,
    ensure_index_exists,
    get_session_by_path,
    save_chunk,
    save_session,
    set_metadata,
)

console = Console()

# Claude Code sessions location
SESSIONS_DIR = Path.home() / ".claude" / "projects"


def discover_sessions() -> list[Path]:
    """Discover all JSONL session files."""
    if not SESSIONS_DIR.exists():
        return []
    return list(SESSIONS_DIR.glob("**/*.jsonl"))


def extract_project_name(path: Path) -> str:
    """Extract project name from session path.

    Path format: ~/.claude/projects/-Users-name-Code-project/session.jsonl
    Returns: project (last component of original path)
    """
    # The parent directory name is the encoded project path
    encoded_path = path.parent.name

    # Decode: -Users-name-Code-project -> /Users/name/Code/project
    # Take the last component as project name
    parts = encoded_path.split("-")

    # Filter out empty parts and common prefixes
    meaningful_parts = [p for p in parts if p and p not in ("Users", "home")]

    if meaningful_parts:
        return meaningful_parts[-1]
    return encoded_path


def parse_session(path: Path) -> tuple[Session, list[Message]] | None:
    """Parse a JSONL session file.

    Returns Session and list of Messages, or None if parsing fails.
    """
    messages: list[Message] = []
    first_timestamp: datetime | None = None
    last_timestamp: datetime | None = None
    session_id: str | None = None

    try:
        with open(path, encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    # Skip invalid JSON lines (may be partial records)
                    continue

                # Skip if not a dict (invalid record)
                if not isinstance(record, dict):
                    continue

                record_type = record.get("type")

                # Extract session ID from first record that has it
                if session_id is None and "sessionId" in record:
                    session_id = record["sessionId"]

                # Parse timestamp
                ts_str = record.get("timestamp")
                if ts_str:
                    try:
                        ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                        if first_timestamp is None:
                            first_timestamp = ts
                        last_timestamp = ts
                    except ValueError:
                        ts = datetime.now(tz=timezone.utc)
                else:
                    ts = datetime.now(tz=timezone.utc)

                if record_type == "user":
                    msg_data = record.get("message", {})
                    content = msg_data.get("content", "")
                    if isinstance(content, str) and content.strip():
                        messages.append(
                            Message(
                                id=record.get("uuid", f"user-{line_num}"),
                                role="user",
                                content=content,
                                content_type="user",
                                timestamp=ts,
                            )
                        )

                elif record_type == "assistant":
                    msg_data = record.get("message", {})
                    content_blocks = msg_data.get("content", [])

                    for block in content_blocks:
                        if not isinstance(block, dict):
                            continue

                        block_type = block.get("type")

                        if block_type == "text":
                            text = block.get("text", "")
                            if text.strip():
                                messages.append(
                                    Message(
                                        id=record.get("uuid", f"asst-{line_num}") + "-text",
                                        role="assistant",
                                        content=text,
                                        content_type="text",
                                        timestamp=ts,
                                    )
                                )

                        elif block_type == "thinking":
                            thinking = block.get("thinking", "")
                            if thinking.strip():
                                messages.append(
                                    Message(
                                        id=record.get("uuid", f"asst-{line_num}") + "-thinking",
                                        role="assistant",
                                        content=thinking,
                                        content_type="thinking",
                                        timestamp=ts,
                                    )
                                )

                        elif block_type == "tool_use":
                            # Include tool name and input as searchable content
                            tool_name = block.get("name", "unknown")
                            tool_input = block.get("input", {})
                            input_json = json.dumps(tool_input, indent=2)
                            content = f"Tool: {tool_name}\nInput: {input_json}"
                            messages.append(
                                Message(
                                    id=record.get("uuid", f"asst-{line_num}") + "-tool",
                                    role="assistant",
                                    content=content,
                                    content_type="tool_use",
                                    timestamp=ts,
                                )
                            )

                        elif block_type == "tool_result":
                            result = block.get("content", "")
                            if isinstance(result, str) and result.strip():
                                messages.append(
                                    Message(
                                        id=record.get("uuid", f"asst-{line_num}") + "-result",
                                        role="assistant",
                                        content=result[:1000],  # Truncate large results
                                        content_type="tool_result",
                                        timestamp=ts,
                                    )
                                )

                # Skip file-history-snapshot and other types

    except Exception:
        # Skip corrupted sessions per spec N4
        return None

    if not messages:
        return None

    # Use path-based unique ID to avoid collisions across directories
    # The filename is already a UUID in most cases, but we make it unique by
    # including the parent directory hash
    unique_id = f"{path.parent.name}_{path.stem}"

    # Ensure we have timestamps
    now = datetime.now(tz=timezone.utc)
    if first_timestamp is None:
        first_timestamp = now
    if last_timestamp is None:
        last_timestamp = now

    session = Session(
        id=unique_id,
        path=path,
        project=extract_project_name(path),
        created_at=first_timestamp,
        updated_at=last_timestamp,
    )

    return session, messages


def filter_messages_for_chunking(messages: list[Message]) -> list[Message]:
    """Filter messages to prioritize user-facing content.

    Per spec:
    - Text content gets full weight
    - Thinking blocks get lower weight
    - Tool use/results get lower weight
    """
    # For chunking, we primarily want text content and user messages
    # We include thinking/tool content but they'll be weighted lower in search
    priority_order = {"user": 0, "text": 1, "thinking": 2, "tool_use": 3, "tool_result": 4}

    return sorted(messages, key=lambda m: (m.timestamp, priority_order.get(m.content_type, 5)))


def build_index(force: bool = False, dry_run: bool = False) -> None:
    """Build or rebuild the search index.

    Args:
        force: Reindex all sessions even if unchanged.
        dry_run: Show what would be indexed without actually indexing.
    """
    conn = ensure_index_exists()

    session_paths = discover_sessions()
    if not session_paths:
        console.print("[yellow]No session files found in ~/.claude/projects[/yellow]")
        return

    console.print(f"Found {len(session_paths)} session files")

    sessions_to_index: list[Path] = []
    for path in session_paths:
        if force:
            sessions_to_index.append(path)
        else:
            # Check if already indexed and not modified
            existing = get_session_by_path(conn, path)
            if existing is None:
                sessions_to_index.append(path)
            else:
                # Check if file was modified
                mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
                if mtime > existing.updated_at:
                    sessions_to_index.append(path)

    if not sessions_to_index:
        console.print("[green]Index is up to date[/green]")
        conn.close()
        return

    if dry_run:
        console.print(f"[yellow]Dry run - would index {len(sessions_to_index)} sessions:[/yellow]")
        # Group by project for cleaner output
        by_project: dict[str, list[Path]] = {}
        for path in sessions_to_index:
            project = extract_project_name(path)
            by_project.setdefault(project, []).append(path)

        for project, paths in sorted(by_project.items()):
            console.print(f"  [cyan]{project}[/cyan]: {len(paths)} sessions")
        conn.close()
        return

    console.print(f"Indexing {len(sessions_to_index)} sessions...")

    # Collect all chunks first, then batch encode
    all_chunks: list[tuple[Session, list[tuple]]] = []  # (session, [(chunk, messages)])

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        parse_task = progress.add_task("Parsing sessions...", total=len(sessions_to_index))

        for path in sessions_to_index:
            result = parse_session(path)
            if result is None:
                progress.advance(parse_task)
                continue

            session, messages = result
            filtered = filter_messages_for_chunking(messages)
            chunks = create_chunks(filtered, session.id)

            if chunks:
                all_chunks.append((session, [(c, messages) for c in chunks]))

            progress.advance(parse_task)

    if not all_chunks:
        console.print("[yellow]No content to index[/yellow]")
        return

    # Flatten chunks for batch encoding
    flat_chunks = [
        (session, chunk, msgs) for session, chunk_list in all_chunks for chunk, msgs in chunk_list
    ]
    texts = [chunk.text for _, chunk, _ in flat_chunks]

    console.print(f"Generating embeddings for {len(texts)} chunks...")
    embeddings = encode_batch(texts)

    console.print("Saving to index...")
    # First, delete existing chunks for sessions we're re-indexing
    deleted_sessions: set[str] = set()
    for session, _ in all_chunks:
        if session.id not in deleted_sessions:
            delete_session_chunks(conn, session.id)
            deleted_sessions.add(session.id)

    # Then save all sessions and chunks
    for (session, chunk, _), embedding in zip(flat_chunks, embeddings, strict=True):
        save_session(conn, session)
        save_chunk(conn, chunk, embedding)

    conn.commit()
    set_metadata(conn, "last_indexed", datetime.now(tz=timezone.utc).isoformat())
    conn.close()

    session_count = len(all_chunks)
    chunk_count = len(flat_chunks)
    console.print(f"[green]Indexed {session_count} sessions with {chunk_count} chunks[/green]")
