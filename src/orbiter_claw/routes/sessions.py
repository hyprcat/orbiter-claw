"""REST API routes for session and message management."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from orbiter_claw.database import get_db
from orbiter_claw.sessions import SessionManager

router = APIRouter(prefix="/sessions", tags=["sessions"])


class CreateSessionRequest(BaseModel):
    title: str = ""
    channel: str = "web"


class UpdateSessionRequest(BaseModel):
    title: str


class AddMessageRequest(BaseModel):
    role: str
    content: str
    metadata: dict[str, object] | None = None


@router.post("")
async def create_session(body: CreateSessionRequest) -> dict[str, object]:
    """Create a new session."""
    async with get_db() as db:
        mgr = SessionManager(db)
        session = await mgr.create_session(title=body.title, channel=body.channel)
        return session.to_dict()


@router.get("")
async def list_sessions(
    q: str | None = Query(default=None, description="Search sessions by title"),
) -> dict[str, object]:
    """List all sessions, optionally filtered by title substring."""
    async with get_db() as db:
        mgr = SessionManager(db)
        sessions = await mgr.list_sessions(q=q)
        return {"sessions": [s.to_dict() for s in sessions]}


@router.get("/{session_id}")
async def get_session(session_id: str) -> dict[str, object]:
    """Get a session by ID."""
    async with get_db() as db:
        mgr = SessionManager(db)
        session = await mgr.get_session(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found")
        return session.to_dict()


@router.put("/{session_id}")
async def update_session(session_id: str, body: UpdateSessionRequest) -> dict[str, object]:
    """Update a session's title."""
    async with get_db() as db:
        mgr = SessionManager(db)
        session = await mgr.update_session(session_id, body.title)
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found")
        return session.to_dict()


@router.delete("/{session_id}")
async def delete_session(session_id: str) -> dict[str, str]:
    """Delete a session and all its messages."""
    async with get_db() as db:
        mgr = SessionManager(db)
        deleted = await mgr.delete_session(session_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Session not found")
        return {"status": "deleted"}


@router.get("/{session_id}/messages")
async def get_messages(
    session_id: str,
    cursor: str | None = Query(default=None, description="Pagination cursor"),
    limit: int = Query(default=50, ge=1, le=200, description="Max messages to return"),
) -> dict[str, object]:
    """Get messages for a session with cursor-based pagination."""
    async with get_db() as db:
        mgr = SessionManager(db)
        # Verify session exists
        session = await mgr.get_session(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found")

        result = await mgr.get_messages(session_id, cursor=cursor, limit=limit)
        return {
            "messages": [m.to_dict() for m in result.messages],
            "next_cursor": result.next_cursor,
            "has_more": result.has_more,
        }


@router.post("/{session_id}/messages")
async def add_message(session_id: str, body: AddMessageRequest) -> dict[str, object]:
    """Add a message to a session. Auto-generates title from first user message."""
    if body.role not in ("user", "assistant", "system", "tool"):
        raise HTTPException(
            status_code=422,
            detail="Invalid role. Must be: user, assistant, system, tool",
        )

    async with get_db() as db:
        mgr = SessionManager(db)
        # Verify session exists
        session = await mgr.get_session(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found")

        message = await mgr.add_message(session_id, body.role, body.content, body.metadata)

        # Auto-title generation from first user message
        if body.role == "user" and not session.title and session.message_count == 0:
            await mgr.auto_title(session_id, body.content)

        return message.to_dict()
