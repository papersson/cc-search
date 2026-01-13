"""Tests for the chunker module."""

from datetime import datetime

from cc_search.chunker import create_chunks, create_sub_chunks, get_content_type_weight
from cc_search.models import Message


def test_create_chunks_basic(sample_messages):
    """Test basic chunk creation from messages."""
    chunks = create_chunks(sample_messages, "test-session")

    assert len(chunks) == 1
    chunk = chunks[0]
    assert "User:" in chunk.text
    assert "Assistant:" in chunk.text
    assert "authentication" in chunk.text.lower()
    assert chunk.session_id == "test-session"


def test_create_chunks_multiple_pairs():
    """Test chunk creation with multiple user-assistant pairs."""
    messages = [
        Message(
            id="1",
            role="user",
            content="Hello",
            content_type="user",
            timestamp=datetime(2024, 1, 1, 10, 0, 0),
        ),
        Message(
            id="2",
            role="assistant",
            content="Hi there!",
            content_type="text",
            timestamp=datetime(2024, 1, 1, 10, 0, 5),
        ),
        Message(
            id="3",
            role="user",
            content="How are you?",
            content_type="user",
            timestamp=datetime(2024, 1, 1, 10, 1, 0),
        ),
        Message(
            id="4",
            role="assistant",
            content="I'm doing well!",
            content_type="text",
            timestamp=datetime(2024, 1, 1, 10, 1, 5),
        ),
    ]

    chunks = create_chunks(messages, "test-session")

    assert len(chunks) == 2
    assert "Hello" in chunks[0].text
    assert "How are you?" in chunks[1].text


def test_create_sub_chunks_short_text():
    """Test that short text doesn't get sub-chunked."""
    text = "This is a short message."
    chunks = create_sub_chunks(
        text=text,
        session_id="test",
        message_ids=["1"],
        content_types={"text"},
        timestamp=datetime.now(),
    )

    assert len(chunks) == 1
    assert chunks[0].text == text


def test_create_sub_chunks_long_text():
    """Test that long text gets sub-chunked."""
    # Create text longer than MAX_CHUNK_CHARS (2000)
    text = "This is a test sentence. " * 200  # ~5000 chars

    chunks = create_sub_chunks(
        text=text,
        session_id="test",
        message_ids=["1"],
        content_types={"text"},
        timestamp=datetime.now(),
    )

    assert len(chunks) > 1
    # All chunks should reference the same messages
    for chunk in chunks:
        assert chunk.message_ids == ["1"]
        assert chunk.session_id == "test"


def test_content_type_weights():
    """Test content type ranking weights per spec N1 and N2."""
    # User-facing content should have highest weight
    assert get_content_type_weight("text") == 1.0
    assert get_content_type_weight("user") == 1.0

    # Thinking blocks should have lower weight (N2)
    assert get_content_type_weight("thinking") < 1.0

    # Tool output should have lower weight (N1)
    assert get_content_type_weight("tool_use") < 1.0
    assert get_content_type_weight("tool_result") < 1.0

    # Thinking should be weighted same or higher than tool output
    assert get_content_type_weight("thinking") >= get_content_type_weight("tool_use")
