"""SQLite storage for cc-search index."""

import contextlib
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

import sqlite_vec

from cc_search.models import Chunk, Session

# Index location
INDEX_DIR = Path.home() / ".local" / "share" / "cc-search"
INDEX_PATH = INDEX_DIR / "index.db"

# Embedding dimension for all-MiniLM-L6-v2
EMBEDDING_DIM = 384


def get_connection() -> sqlite3.Connection:
    """Get a connection to the index database."""
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(INDEX_PATH))
    conn.row_factory = sqlite3.Row
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    """Initialize the database schema."""
    conn.executescript("""
        -- Sessions table
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            path TEXT NOT NULL,
            project TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        -- Chunks table
        CREATE TABLE IF NOT EXISTS chunks (
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL REFERENCES sessions(id),
            message_ids TEXT NOT NULL,  -- JSON array
            text TEXT NOT NULL,
            content_types TEXT NOT NULL,  -- JSON array
            timestamp TEXT NOT NULL
        );

        -- FTS5 for keyword search
        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
            text,
            content='chunks',
            content_rowid='rowid'
        );

        -- Triggers to keep FTS in sync
        CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
            INSERT INTO chunks_fts(rowid, text) VALUES (new.rowid, new.text);
        END;

        CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
            INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES('delete', old.rowid, old.text);
        END;

        CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
            INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES('delete', old.rowid, old.text);
            INSERT INTO chunks_fts(rowid, text) VALUES (new.rowid, new.text);
        END;

        -- Metadata table for tracking index state
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
    """)

    # Create vector table (sqlite-vec syntax)
    with contextlib.suppress(sqlite3.OperationalError):
        conn.execute(f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_vec USING vec0(
                chunk_id TEXT PRIMARY KEY,
                embedding float[{EMBEDDING_DIM}]
            );
        """)

    conn.commit()


def ensure_index_exists() -> sqlite3.Connection:
    """Ensure the index database exists and is initialized."""
    conn = get_connection()
    init_schema(conn)
    return conn


def index_exists() -> bool:
    """Check if the index database exists."""
    return INDEX_PATH.exists()


def save_session(conn: sqlite3.Connection, session: Session) -> None:
    """Save or update a session."""
    conn.execute(
        """
        INSERT OR REPLACE INTO sessions (id, path, project, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            session.id,
            str(session.path),
            session.project,
            session.created_at.isoformat(),
            session.updated_at.isoformat(),
        ),
    )


def save_chunk(conn: sqlite3.Connection, chunk: Chunk, embedding: list[float]) -> None:
    """Save a chunk with its embedding."""
    conn.execute(
        """
        INSERT OR REPLACE INTO chunks (id, session_id, message_ids, text, content_types, timestamp)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            chunk.id,
            chunk.session_id,
            json.dumps(chunk.message_ids),
            chunk.text,
            json.dumps(list(chunk.content_types)),
            chunk.timestamp.isoformat(),
        ),
    )

    # Insert embedding
    conn.execute(
        """
        INSERT OR REPLACE INTO chunks_vec (chunk_id, embedding)
        VALUES (?, ?)
        """,
        (chunk.id, sqlite_vec.serialize_float32(embedding)),
    )


def delete_session_chunks(conn: sqlite3.Connection, session_id: str) -> None:
    """Delete all chunks for a session."""
    # Get chunk IDs first for vector table cleanup
    chunk_ids = [
        row[0] for row in conn.execute("SELECT id FROM chunks WHERE session_id = ?", (session_id,))
    ]

    # Delete from vector table
    for chunk_id in chunk_ids:
        conn.execute("DELETE FROM chunks_vec WHERE chunk_id = ?", (chunk_id,))

    # Delete chunks (triggers will clean up FTS)
    conn.execute("DELETE FROM chunks WHERE session_id = ?", (session_id,))


def get_session(conn: sqlite3.Connection, session_id: str) -> Session | None:
    """Get a session by ID."""
    row = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
    if row is None:
        return None
    return Session(
        id=row["id"],
        path=Path(row["path"]),
        project=row["project"],
        created_at=datetime.fromisoformat(row["created_at"]),
        updated_at=datetime.fromisoformat(row["updated_at"]),
    )


def get_session_by_path(conn: sqlite3.Connection, path: Path) -> Session | None:
    """Get a session by file path."""
    row = conn.execute("SELECT * FROM sessions WHERE path = ?", (str(path),)).fetchone()
    if row is None:
        return None
    return Session(
        id=row["id"],
        path=Path(row["path"]),
        project=row["project"],
        created_at=datetime.fromisoformat(row["created_at"]),
        updated_at=datetime.fromisoformat(row["updated_at"]),
    )


def get_chunk(conn: sqlite3.Connection, chunk_id: str) -> Chunk | None:
    """Get a chunk by ID."""
    row = conn.execute("SELECT * FROM chunks WHERE id = ?", (chunk_id,)).fetchone()
    if row is None:
        return None
    return Chunk(
        id=row["id"],
        session_id=row["session_id"],
        message_ids=json.loads(row["message_ids"]),
        text=row["text"],
        content_types=set(json.loads(row["content_types"])),
        timestamp=datetime.fromisoformat(row["timestamp"]),
    )


def get_chunk_by_prefix(conn: sqlite3.Connection, prefix: str) -> Chunk | None:
    """Get a chunk by ID prefix (first 8+ chars)."""
    row = conn.execute(
        "SELECT * FROM chunks WHERE id LIKE ? LIMIT 1", (f"{prefix}%",)
    ).fetchone()
    if row is None:
        return None
    return Chunk(
        id=row["id"],
        session_id=row["session_id"],
        message_ids=json.loads(row["message_ids"]),
        text=row["text"],
        content_types=set(json.loads(row["content_types"])),
        timestamp=datetime.fromisoformat(row["timestamp"]),
    )


def get_chunks_by_message_ids(conn: sqlite3.Connection, message_ids: list[str]) -> list[Chunk]:
    """Get all chunks with the same message_ids (for reconstructing sub-chunked content)."""
    message_ids_json = json.dumps(message_ids)
    rows = conn.execute(
        "SELECT * FROM chunks WHERE message_ids = ?", (message_ids_json,)
    ).fetchall()
    return [
        Chunk(
            id=row["id"],
            session_id=row["session_id"],
            message_ids=json.loads(row["message_ids"]),
            text=row["text"],
            content_types=set(json.loads(row["content_types"])),
            timestamp=datetime.fromisoformat(row["timestamp"]),
        )
        for row in rows
    ]


def set_metadata(conn: sqlite3.Connection, key: str, value: str) -> None:
    """Set a metadata value."""
    conn.execute(
        "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
        (key, value),
    )
    conn.commit()


def get_metadata(conn: sqlite3.Connection, key: str) -> str | None:
    """Get a metadata value."""
    row = conn.execute("SELECT value FROM metadata WHERE key = ?", (key,)).fetchone()
    return row["value"] if row else None


def get_index_stats() -> dict[str, Any]:
    """Get index statistics."""
    if not index_exists():
        return {
            "session_count": 0,
            "chunk_count": 0,
            "index_path": str(INDEX_PATH),
            "last_indexed": None,
        }

    conn = get_connection()
    session_count = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
    chunk_count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    last_indexed = get_metadata(conn, "last_indexed")
    conn.close()

    return {
        "session_count": session_count,
        "chunk_count": chunk_count,
        "index_path": str(INDEX_PATH),
        "last_indexed": last_indexed,
    }


def get_all_projects(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """Get all unique projects with their session counts."""
    rows = conn.execute("""
        SELECT project, COUNT(*) as session_count, MAX(updated_at) as last_updated
        FROM sessions
        GROUP BY project
        ORDER BY last_updated DESC
    """).fetchall()
    return [
        {"project": row["project"], "sessions": row["session_count"], "last_updated": row["last_updated"]}
        for row in rows
    ]


def get_detailed_stats(conn: sqlite3.Connection) -> dict[str, Any]:
    """Get detailed index statistics with per-project breakdown."""
    # Per-project stats
    project_stats = conn.execute("""
        SELECT
            s.project,
            COUNT(DISTINCT s.id) as sessions,
            COUNT(c.id) as chunks,
            MIN(s.created_at) as oldest,
            MAX(s.updated_at) as newest
        FROM sessions s
        LEFT JOIN chunks c ON s.id = c.session_id
        GROUP BY s.project
        ORDER BY COUNT(c.id) DESC
    """).fetchall()

    # Index file size
    index_size = INDEX_PATH.stat().st_size if INDEX_PATH.exists() else 0

    return {
        "projects": [dict(row) for row in project_stats],
        "index_size_bytes": index_size,
        "index_size_human": _format_size(index_size),
    }


def _format_size(size_bytes: int) -> str:
    """Format byte size as human-readable string."""
    size = float(size_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def get_session_chunks_ordered(conn: sqlite3.Connection, session_id: str) -> list[Chunk]:
    """Get all chunks for a session ordered by timestamp."""
    rows = conn.execute(
        "SELECT * FROM chunks WHERE session_id = ? ORDER BY timestamp",
        (session_id,)
    ).fetchall()
    return [
        Chunk(
            id=row["id"],
            session_id=row["session_id"],
            message_ids=json.loads(row["message_ids"]),
            text=row["text"],
            content_types=set(json.loads(row["content_types"])),
            timestamp=datetime.fromisoformat(row["timestamp"]),
        )
        for row in rows
    ]
