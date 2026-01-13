"""Data models for cc-search."""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass
class Session:
    """A Claude Code session (JSONL file)."""

    id: str
    path: Path
    project: str
    created_at: datetime
    updated_at: datetime


@dataclass
class Message:
    """A single message within a session."""

    id: str
    role: str  # "user" | "assistant"
    content: str
    content_type: str  # "text" | "thinking" | "tool_use" | "tool_result"
    timestamp: datetime


@dataclass
class Chunk:
    """A searchable unit (user+assistant pair or sub-chunk)."""

    id: str
    session_id: str
    message_ids: list[str]
    text: str
    content_types: set[str] = field(default_factory=set)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class SearchResult:
    """A search result with context."""

    chunk: Chunk
    score: float
    session: Session
    context_messages: list[Message] = field(default_factory=list)
