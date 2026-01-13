"""Chunking logic for session messages."""

import uuid
from datetime import datetime

from cc_search.models import Chunk, Message

# Approximate token limit for MiniLM
MAX_CHUNK_CHARS = 2000  # ~500 tokens assuming 4 chars/token


def estimate_tokens(text: str) -> int:
    """Estimate token count (rough approximation)."""
    return len(text) // 4


def create_chunks(messages: list[Message], session_id: str) -> list[Chunk]:
    """Create chunks from a list of messages.

    Primary strategy: pair user messages with following assistant responses.
    Sub-chunk if the pair exceeds token limits.
    """
    chunks: list[Chunk] = []

    i = 0
    while i < len(messages):
        msg = messages[i]

        if msg.role == "user":
            # Find all following assistant messages until next user message
            user_msg = msg
            assistant_msgs: list[Message] = []

            j = i + 1
            while j < len(messages) and messages[j].role == "assistant":
                assistant_msgs.append(messages[j])
                j += 1

            # Combine into chunk text
            chunk_text = f"User: {user_msg.content}"
            message_ids = [user_msg.id]
            content_types = {user_msg.content_type}
            timestamp = user_msg.timestamp

            for asst_msg in assistant_msgs:
                chunk_text += f"\n\nAssistant: {asst_msg.content}"
                message_ids.append(asst_msg.id)
                content_types.add(asst_msg.content_type)
                # Use latest timestamp
                if asst_msg.timestamp > timestamp:
                    timestamp = asst_msg.timestamp

            # Create chunk(s), potentially sub-chunking if too long
            sub_chunks = create_sub_chunks(
                text=chunk_text,
                session_id=session_id,
                message_ids=message_ids,
                content_types=content_types,
                timestamp=timestamp,
            )
            chunks.extend(sub_chunks)

            i = j  # Skip past the assistant messages
        else:
            # Orphan assistant message (no preceding user message)
            # Create a chunk for it anyway
            chunk = Chunk(
                id=str(uuid.uuid4()),
                session_id=session_id,
                message_ids=[msg.id],
                text=f"Assistant: {msg.content}",
                content_types={msg.content_type},
                timestamp=msg.timestamp,
            )
            chunks.append(chunk)
            i += 1

    return chunks


def create_sub_chunks(
    text: str,
    session_id: str,
    message_ids: list[str],
    content_types: set[str],
    timestamp: datetime,
) -> list[Chunk]:
    """Sub-chunk text if it exceeds the max token limit."""
    if len(text) <= MAX_CHUNK_CHARS:
        return [
            Chunk(
                id=str(uuid.uuid4()),
                session_id=session_id,
                message_ids=message_ids,
                text=text,
                content_types=content_types,
                timestamp=timestamp,
            )
        ]

    # Split into overlapping chunks
    chunks: list[Chunk] = []
    overlap = MAX_CHUNK_CHARS // 4  # 25% overlap
    start = 0

    while start < len(text):
        end = start + MAX_CHUNK_CHARS

        # Try to break at a sentence or paragraph boundary
        if end < len(text):
            # Look for paragraph break first
            para_break = text.rfind("\n\n", start + overlap, end)
            if para_break > start + overlap:
                end = para_break
            else:
                # Look for sentence break
                sentence_break = text.rfind(". ", start + overlap, end)
                if sentence_break > start + overlap:
                    end = sentence_break + 1

        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append(
                Chunk(
                    id=str(uuid.uuid4()),
                    session_id=session_id,
                    message_ids=message_ids,
                    text=chunk_text,
                    content_types=content_types,
                    timestamp=timestamp,
                )
            )

        # Move start, accounting for overlap
        start = end - overlap if end < len(text) else len(text)

    return chunks


def get_content_type_weight(content_type: str) -> float:
    """Get the ranking weight for a content type.

    Per spec constraints N1 and N2:
    - N1: MUST NOT rank tool output above discussion by default
    - N2: MUST NOT rank thinking blocks above user-facing content
    """
    weights = {
        "text": 1.0,
        "user": 1.0,
        "thinking": 0.5,
        "tool_use": 0.3,
        "tool_result": 0.3,
    }
    return weights.get(content_type, 0.5)
