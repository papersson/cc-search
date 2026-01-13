"""Pytest fixtures for cc-search tests."""

import json
import tempfile
from datetime import datetime
from pathlib import Path

import pytest


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_session_jsonl(temp_dir):
    """Create a sample JSONL session file."""
    session_file = temp_dir / "test-session.jsonl"

    records = [
        {
            "type": "user",
            "uuid": "msg-001",
            "sessionId": "test-session-123",
            "timestamp": "2024-01-15T10:00:00Z",
            "message": {"role": "user", "content": "How do I implement authentication?"},
        },
        {
            "type": "assistant",
            "uuid": "msg-002",
            "sessionId": "test-session-123",
            "timestamp": "2024-01-15T10:00:05Z",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "For authentication, you can use JWT tokens..."},
                ],
            },
        },
        {
            "type": "user",
            "uuid": "msg-003",
            "sessionId": "test-session-123",
            "timestamp": "2024-01-15T10:01:00Z",
            "message": {"role": "user", "content": "Can you show me an example?"},
        },
        {
            "type": "assistant",
            "uuid": "msg-004",
            "sessionId": "test-session-123",
            "timestamp": "2024-01-15T10:01:10Z",
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "type": "thinking",
                        "thinking": "Let me think about a good example...",
                    },
                    {
                        "type": "text",
                        "text": "Here's an example of JWT authentication in Python...",
                    },
                ],
            },
        },
    ]

    with open(session_file, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    return session_file


@pytest.fixture
def sample_messages():
    """Create sample Message objects for testing."""
    from cc_search.models import Message

    return [
        Message(
            id="msg-001",
            role="user",
            content="How do I implement authentication?",
            content_type="user",
            timestamp=datetime(2024, 1, 15, 10, 0, 0),
        ),
        Message(
            id="msg-002",
            role="assistant",
            content="For authentication, you can use JWT tokens...",
            content_type="text",
            timestamp=datetime(2024, 1, 15, 10, 0, 5),
        ),
    ]
