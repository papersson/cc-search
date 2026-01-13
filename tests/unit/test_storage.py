"""Tests for the storage module."""

import sqlite3
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest

from cc_search.models import Chunk, Session
from cc_search.storage import (
    ensure_index_exists,
    get_chunk,
    get_session,
    init_schema,
    save_chunk,
    save_session,
)


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        with patch("cc_search.storage.INDEX_PATH", db_path):
            with patch("cc_search.storage.INDEX_DIR", Path(tmpdir)):
                conn = ensure_index_exists()
                yield conn, db_path
                conn.close()


def test_init_schema(temp_db):
    """Test schema initialization creates all tables."""
    conn, _ = temp_db

    # Check tables exist
    tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    table_names = {t[0] for t in tables}

    assert "sessions" in table_names
    assert "chunks" in table_names
    assert "metadata" in table_names


def test_save_and_get_session(temp_db):
    """Test saving and retrieving a session."""
    conn, _ = temp_db

    session = Session(
        id="test-123",
        path=Path("/test/session.jsonl"),
        project="test-project",
        created_at=datetime(2024, 1, 15, 10, 0, 0),
        updated_at=datetime(2024, 1, 15, 11, 0, 0),
    )

    save_session(conn, session)
    conn.commit()

    retrieved = get_session(conn, "test-123")
    assert retrieved is not None
    assert retrieved.id == session.id
    assert retrieved.project == session.project
    assert retrieved.path == session.path


def test_save_and_get_chunk(temp_db):
    """Test saving and retrieving a chunk with embedding."""
    conn, _ = temp_db

    # First save a session
    session = Session(
        id="session-1",
        path=Path("/test/session.jsonl"),
        project="test",
        created_at=datetime.now(),
        updated_at=datetime.now(),
    )
    save_session(conn, session)

    # Then save a chunk
    chunk = Chunk(
        id="chunk-1",
        session_id="session-1",
        message_ids=["msg-1", "msg-2"],
        text="Test chunk content",
        content_types={"text", "user"},
        timestamp=datetime.now(),
    )

    # Create a dummy embedding (384 dimensions for MiniLM)
    embedding = [0.1] * 384

    save_chunk(conn, chunk, embedding)
    conn.commit()

    retrieved = get_chunk(conn, "chunk-1")
    assert retrieved is not None
    assert retrieved.id == chunk.id
    assert retrieved.text == chunk.text
    assert retrieved.session_id == chunk.session_id
    assert "text" in retrieved.content_types


def test_session_not_found(temp_db):
    """Test getting non-existent session returns None."""
    conn, _ = temp_db
    assert get_session(conn, "nonexistent") is None


def test_chunk_not_found(temp_db):
    """Test getting non-existent chunk returns None."""
    conn, _ = temp_db
    assert get_chunk(conn, "nonexistent") is None
