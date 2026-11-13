"""Session and conversation management with persistent SQLite storage.

Message content and metadata are transparently encrypted/decrypted via the
database module's encrypt_value/decrypt_value functions.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass

import aiosqlite

from orbiter_claw.database import decrypt_value, encrypt_value


@dataclass
class Message:
    """A single message within a session."""

    id: str
    session_id: str
    role: str  # user, assistant, system, tool
    content: str
    metadata: dict[str, object]
    created_at: float

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "session_id": self.session_id,
            "role": self.role,
            "content": self.content,
            "metadata": self.metadata,
            "created_at": self.created_at,
        }


@dataclass
class Session:
    """A conversation session with metadata."""

    id: str
    title: str
    channel: str
    created_at: float
    updated_at: float
    message_count: int = 0

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "title": self.title,
            "channel": self.channel,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "message_count": self.message_count,
        }


@dataclass
class PaginatedMessages:
    """Cursor-based pagination result for messages."""

    messages: list[Message]
    next_cursor: str | None
    has_more: bool


def _row_to_session(row: aiosqlite.Row) -> Session:
    """Convert a database row to a Session object."""
    return Session(
        id=row["id"],
        title=row["title"],
        channel=row["channel"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        message_count=row["message_count"],
    )


def _row_to_message(row: aiosqlite.Row) -> Message:
    """Convert a database row to a Message object, decrypting content and metadata."""
    raw_content = row["content"]
    raw_metadata = row["metadata"]
    return Message(
        id=row["id"],
        session_id=row["session_id"],
        role=row["role"],
        content=decrypt_value(raw_content),
        metadata=json.loads(decrypt_value(raw_metadata)) if raw_metadata else {},
        created_at=row["created_at"],
    )


class SessionManager:
    """Manages session and message CRUD operations."""

    def __init__(self, db: aiosqlite.Connection) -> None:
        self._db = db

    async def create_session(
        self,
        title: str = "",
        channel: str = "web",
    ) -> Session:
        """Create a new session."""
        session_id = str(uuid.uuid4())
        now = time.time()
        await self._db.execute(
            "INSERT INTO sessions (id, title, channel, created_at, updated_at)"
            " VALUES (?, ?, ?, ?, ?)",
            (session_id, title, channel, now, now),
        )
        await self._db.commit()
        return Session(
            id=session_id,
            title=title,
            channel=channel,
            created_at=now,
            updated_at=now,
            message_count=0,
        )

    async def get_session(self, session_id: str) -> Session | None:
        """Get a session by ID with message count."""
        cursor = await self._db.execute(
            """
            SELECT s.*, COUNT(m.id) as message_count
            FROM sessions s
            LEFT JOIN messages m ON m.session_id = s.id
            WHERE s.id = ?
            GROUP BY s.id
            """,
            (session_id,),
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return _row_to_session(row)

    async def list_sessions(self, q: str | None = None) -> list[Session]:
        """List all sessions, optionally filtered by title substring."""
        if q:
            cursor = await self._db.execute(
                """
                SELECT s.*, COUNT(m.id) as message_count
                FROM sessions s
                LEFT JOIN messages m ON m.session_id = s.id
                WHERE s.title LIKE ?
                GROUP BY s.id
                ORDER BY s.updated_at DESC
                """,
                (f"%{q}%",),
            )
        else:
            cursor = await self._db.execute(
                """
                SELECT s.*, COUNT(m.id) as message_count
                FROM sessions s
                LEFT JOIN messages m ON m.session_id = s.id
                GROUP BY s.id
                ORDER BY s.updated_at DESC
                """
            )
        rows = await cursor.fetchall()
        return [_row_to_session(row) for row in rows]

    async def update_session(self, session_id: str, title: str) -> Session | None:
        """Update a session's title."""
        now = time.time()
        cursor = await self._db.execute(
            "UPDATE sessions SET title = ?, updated_at = ? WHERE id = ?",
            (title, now, session_id),
        )
        await self._db.commit()
        if cursor.rowcount == 0:
            return None
        return await self.get_session(session_id)

    async def delete_session(self, session_id: str) -> bool:
        """Delete a session and its messages (CASCADE)."""
        cursor = await self._db.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        await self._db.commit()
        return cursor.rowcount > 0

    async def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: dict[str, object] | None = None,
    ) -> Message:
        """Add a message to a session. Content and metadata are encrypted at rest."""
        message_id = str(uuid.uuid4())
        now = time.time()
        meta_json = json.dumps(metadata or {})
        encrypted_content = encrypt_value(content)
        encrypted_metadata = encrypt_value(meta_json)
        await self._db.execute(
            "INSERT INTO messages (id, session_id, role, content, metadata, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (message_id, session_id, role, encrypted_content, encrypted_metadata, now),
        )
        # Update session updated_at
        await self._db.execute("UPDATE sessions SET updated_at = ? WHERE id = ?", (now, session_id))
        await self._db.commit()
        return Message(
            id=message_id,
            session_id=session_id,
            role=role,
            content=content,
            metadata=metadata or {},
            created_at=now,
        )

    async def get_messages(
        self,
        session_id: str,
        cursor: str | None = None,
        limit: int = 50,
    ) -> PaginatedMessages:
        """Get messages for a session with cursor-based pagination.

        Cursor is a created_at timestamp. Messages are returned oldest-first.
        """
        if limit < 1:
            limit = 1
        if limit > 200:
            limit = 200

        # Fetch one extra to detect has_more
        fetch_limit = limit + 1

        if cursor:
            cursor_ts = float(cursor)
            db_cursor = await self._db.execute(
                """
                SELECT * FROM messages
                WHERE session_id = ? AND created_at > ?
                ORDER BY created_at ASC
                LIMIT ?
                """,
                (session_id, cursor_ts, fetch_limit),
            )
        else:
            db_cursor = await self._db.execute(
                """
                SELECT * FROM messages
                WHERE session_id = ?
                ORDER BY created_at ASC
                LIMIT ?
                """,
                (session_id, fetch_limit),
            )

        rows = await db_cursor.fetchall()
        messages = [_row_to_message(row) for row in rows]

        has_more = len(messages) > limit
        if has_more:
            messages = messages[:limit]

        next_cursor = str(messages[-1].created_at) if has_more and messages else None

        return PaginatedMessages(
            messages=messages,
            next_cursor=next_cursor,
            has_more=has_more,
        )

    async def auto_title(self, session_id: str, first_message: str) -> str:
        """Generate a title from the first user message.

        Uses a simple truncation approach. When US-003 (agent runtime) is
        available, this can be upgraded to LLM summarization.
        """
        title = first_message.strip()
        if len(title) > 80:
            title = title[:77] + "..."
        if title:
            await self.update_session(session_id, title)
        return title
