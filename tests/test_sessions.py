"""Tests for session and conversation management (US-002)."""

from __future__ import annotations

from httpx import AsyncClient

from orbiter_claw.database import get_db
from orbiter_claw.sessions import SessionManager

# ---------------------------------------------------------------------------
# Unit tests: SessionManager (direct database operations)
# ---------------------------------------------------------------------------


class TestSessionManager:
    async def test_create_session(self) -> None:
        async with get_db() as db:
            mgr = SessionManager(db)
            session = await mgr.create_session(title="Test", channel="web")
            assert session.id
            assert session.title == "Test"
            assert session.channel == "web"
            assert session.message_count == 0

    async def test_create_session_defaults(self) -> None:
        async with get_db() as db:
            mgr = SessionManager(db)
            session = await mgr.create_session()
            assert session.title == ""
            assert session.channel == "web"

    async def test_get_session(self) -> None:
        async with get_db() as db:
            mgr = SessionManager(db)
            created = await mgr.create_session(title="Lookup")
            fetched = await mgr.get_session(created.id)
            assert fetched is not None
            assert fetched.id == created.id
            assert fetched.title == "Lookup"

    async def test_get_session_not_found(self) -> None:
        async with get_db() as db:
            mgr = SessionManager(db)
            result = await mgr.get_session("nonexistent-id")
            assert result is None

    async def test_list_sessions_empty(self) -> None:
        async with get_db() as db:
            mgr = SessionManager(db)
            sessions = await mgr.list_sessions()
            assert sessions == []

    async def test_list_sessions_ordered_by_updated(self) -> None:
        async with get_db() as db:
            mgr = SessionManager(db)
            s1 = await mgr.create_session(title="First")
            await mgr.create_session(title="Second")
            # Update s1 so it becomes most recent
            await mgr.update_session(s1.id, "First Updated")
            sessions = await mgr.list_sessions()
            assert len(sessions) == 2
            assert sessions[0].id == s1.id  # most recently updated

    async def test_list_sessions_search(self) -> None:
        async with get_db() as db:
            mgr = SessionManager(db)
            await mgr.create_session(title="Python tutorial")
            await mgr.create_session(title="Rust guide")
            await mgr.create_session(title="Python async")
            results = await mgr.list_sessions(q="Python")
            assert len(results) == 2

    async def test_list_sessions_search_no_match(self) -> None:
        async with get_db() as db:
            mgr = SessionManager(db)
            await mgr.create_session(title="Python tutorial")
            results = await mgr.list_sessions(q="JavaScript")
            assert len(results) == 0

    async def test_update_session(self) -> None:
        async with get_db() as db:
            mgr = SessionManager(db)
            session = await mgr.create_session(title="Old")
            updated = await mgr.update_session(session.id, "New Title")
            assert updated is not None
            assert updated.title == "New Title"
            assert updated.updated_at >= session.updated_at

    async def test_update_session_not_found(self) -> None:
        async with get_db() as db:
            mgr = SessionManager(db)
            result = await mgr.update_session("nonexistent", "Title")
            assert result is None

    async def test_delete_session(self) -> None:
        async with get_db() as db:
            mgr = SessionManager(db)
            session = await mgr.create_session(title="ToDelete")
            deleted = await mgr.delete_session(session.id)
            assert deleted is True
            assert await mgr.get_session(session.id) is None

    async def test_delete_session_not_found(self) -> None:
        async with get_db() as db:
            mgr = SessionManager(db)
            deleted = await mgr.delete_session("nonexistent")
            assert deleted is False

    async def test_delete_session_cascades_messages(self) -> None:
        async with get_db() as db:
            mgr = SessionManager(db)
            session = await mgr.create_session(title="Cascade")
            await mgr.add_message(session.id, "user", "hello")
            await mgr.add_message(session.id, "assistant", "hi")
            await mgr.delete_session(session.id)
            # Messages should be gone too
            result = await mgr.get_messages(session.id)
            assert len(result.messages) == 0


class TestMessages:
    async def test_add_message(self) -> None:
        async with get_db() as db:
            mgr = SessionManager(db)
            session = await mgr.create_session()
            msg = await mgr.add_message(session.id, "user", "Hello!")
            assert msg.id
            assert msg.session_id == session.id
            assert msg.role == "user"
            assert msg.content == "Hello!"
            assert msg.metadata == {}

    async def test_add_message_with_metadata(self) -> None:
        async with get_db() as db:
            mgr = SessionManager(db)
            session = await mgr.create_session()
            meta = {"model": "gpt-4", "tokens": 42}
            msg = await mgr.add_message(session.id, "assistant", "Hi", metadata=meta)
            assert msg.metadata == meta

    async def test_add_message_updates_session(self) -> None:
        async with get_db() as db:
            mgr = SessionManager(db)
            session = await mgr.create_session()
            original_updated = session.updated_at
            # small sleep to ensure time difference
            await mgr.add_message(session.id, "user", "Hello!")
            refreshed = await mgr.get_session(session.id)
            assert refreshed is not None
            assert refreshed.updated_at >= original_updated
            assert refreshed.message_count == 1

    async def test_get_messages_empty(self) -> None:
        async with get_db() as db:
            mgr = SessionManager(db)
            session = await mgr.create_session()
            result = await mgr.get_messages(session.id)
            assert result.messages == []
            assert result.next_cursor is None
            assert result.has_more is False

    async def test_get_messages_ordered(self) -> None:
        async with get_db() as db:
            mgr = SessionManager(db)
            session = await mgr.create_session()
            await mgr.add_message(session.id, "user", "first")
            await mgr.add_message(session.id, "assistant", "second")
            await mgr.add_message(session.id, "user", "third")
            result = await mgr.get_messages(session.id)
            assert len(result.messages) == 3
            assert result.messages[0].content == "first"
            assert result.messages[1].content == "second"
            assert result.messages[2].content == "third"

    async def test_cursor_pagination(self) -> None:
        async with get_db() as db:
            mgr = SessionManager(db)
            session = await mgr.create_session()
            for i in range(5):
                await mgr.add_message(session.id, "user", f"msg-{i}")

            # Get first 2
            page1 = await mgr.get_messages(session.id, limit=2)
            assert len(page1.messages) == 2
            assert page1.has_more is True
            assert page1.next_cursor is not None
            assert page1.messages[0].content == "msg-0"
            assert page1.messages[1].content == "msg-1"

            # Get next 2
            page2 = await mgr.get_messages(session.id, cursor=page1.next_cursor, limit=2)
            assert len(page2.messages) == 2
            assert page2.has_more is True
            assert page2.next_cursor is not None
            assert page2.messages[0].content == "msg-2"
            assert page2.messages[1].content == "msg-3"

            # Get last page
            page3 = await mgr.get_messages(session.id, cursor=page2.next_cursor, limit=2)
            assert len(page3.messages) == 1
            assert page3.has_more is False
            assert page3.next_cursor is None
            assert page3.messages[0].content == "msg-4"

    async def test_pagination_limit_clamped(self) -> None:
        async with get_db() as db:
            mgr = SessionManager(db)
            session = await mgr.create_session()
            await mgr.add_message(session.id, "user", "msg")
            # limit < 1 gets clamped to 1
            result = await mgr.get_messages(session.id, limit=0)
            assert len(result.messages) == 1

    async def test_message_to_dict(self) -> None:
        async with get_db() as db:
            mgr = SessionManager(db)
            session = await mgr.create_session()
            msg = await mgr.add_message(session.id, "user", "Hello!")
            d = msg.to_dict()
            assert d["id"] == msg.id
            assert d["role"] == "user"
            assert d["content"] == "Hello!"
            assert d["session_id"] == session.id
            assert "created_at" in d
            assert "metadata" in d


class TestAutoTitle:
    async def test_auto_title_from_short_message(self) -> None:
        async with get_db() as db:
            mgr = SessionManager(db)
            session = await mgr.create_session()
            title = await mgr.auto_title(session.id, "How do I use Python asyncio?")
            assert title == "How do I use Python asyncio?"
            updated = await mgr.get_session(session.id)
            assert updated is not None
            assert updated.title == title

    async def test_auto_title_truncates_long_message(self) -> None:
        async with get_db() as db:
            mgr = SessionManager(db)
            session = await mgr.create_session()
            long_msg = "A" * 100
            title = await mgr.auto_title(session.id, long_msg)
            assert len(title) == 80
            assert title.endswith("...")

    async def test_auto_title_strips_whitespace(self) -> None:
        async with get_db() as db:
            mgr = SessionManager(db)
            session = await mgr.create_session()
            title = await mgr.auto_title(session.id, "  hello  ")
            assert title == "hello"


# ---------------------------------------------------------------------------
# API integration tests
# ---------------------------------------------------------------------------


class TestSessionAPI:
    async def test_create_session_api(self, client: AsyncClient) -> None:
        resp = await client.post("/api/v1/sessions", json={"title": "My Chat"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["title"] == "My Chat"
        assert data["channel"] == "web"
        assert "id" in data

    async def test_create_session_defaults_api(self, client: AsyncClient) -> None:
        resp = await client.post("/api/v1/sessions", json={})
        assert resp.status_code == 200
        data = resp.json()
        assert data["title"] == ""
        assert data["channel"] == "web"

    async def test_list_sessions_empty_api(self, client: AsyncClient) -> None:
        resp = await client.get("/api/v1/sessions")
        assert resp.status_code == 200
        assert resp.json() == {"sessions": []}

    async def test_list_sessions_api(self, client: AsyncClient) -> None:
        await client.post("/api/v1/sessions", json={"title": "One"})
        await client.post("/api/v1/sessions", json={"title": "Two"})
        resp = await client.get("/api/v1/sessions")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["sessions"]) == 2

    async def test_get_session_api(self, client: AsyncClient) -> None:
        create = await client.post("/api/v1/sessions", json={"title": "Fetch Me"})
        sid = create.json()["id"]
        resp = await client.get(f"/api/v1/sessions/{sid}")
        assert resp.status_code == 200
        assert resp.json()["title"] == "Fetch Me"

    async def test_get_session_not_found_api(self, client: AsyncClient) -> None:
        resp = await client.get("/api/v1/sessions/nonexistent")
        assert resp.status_code == 404

    async def test_update_session_api(self, client: AsyncClient) -> None:
        create = await client.post("/api/v1/sessions", json={"title": "Old"})
        sid = create.json()["id"]
        resp = await client.put(f"/api/v1/sessions/{sid}", json={"title": "New"})
        assert resp.status_code == 200
        assert resp.json()["title"] == "New"

    async def test_update_session_not_found_api(self, client: AsyncClient) -> None:
        resp = await client.put("/api/v1/sessions/nonexistent", json={"title": "X"})
        assert resp.status_code == 404

    async def test_delete_session_api(self, client: AsyncClient) -> None:
        create = await client.post("/api/v1/sessions", json={"title": "Delete Me"})
        sid = create.json()["id"]
        resp = await client.delete(f"/api/v1/sessions/{sid}")
        assert resp.status_code == 200
        assert resp.json() == {"status": "deleted"}
        # Confirm gone
        resp = await client.get(f"/api/v1/sessions/{sid}")
        assert resp.status_code == 404

    async def test_delete_session_not_found_api(self, client: AsyncClient) -> None:
        resp = await client.delete("/api/v1/sessions/nonexistent")
        assert resp.status_code == 404

    async def test_search_sessions_api(self, client: AsyncClient) -> None:
        await client.post("/api/v1/sessions", json={"title": "Python tutorial"})
        await client.post("/api/v1/sessions", json={"title": "Rust guide"})
        resp = await client.get("/api/v1/sessions", params={"q": "Python"})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["sessions"]) == 1
        assert data["sessions"][0]["title"] == "Python tutorial"


class TestMessageAPI:
    async def test_add_message_api(self, client: AsyncClient) -> None:
        create = await client.post("/api/v1/sessions", json={})
        sid = create.json()["id"]
        resp = await client.post(
            f"/api/v1/sessions/{sid}/messages",
            json={"role": "user", "content": "Hello!"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["role"] == "user"
        assert data["content"] == "Hello!"
        assert data["session_id"] == sid

    async def test_add_message_invalid_role_api(self, client: AsyncClient) -> None:
        create = await client.post("/api/v1/sessions", json={})
        sid = create.json()["id"]
        resp = await client.post(
            f"/api/v1/sessions/{sid}/messages",
            json={"role": "invalid", "content": "test"},
        )
        assert resp.status_code == 422

    async def test_add_message_session_not_found_api(self, client: AsyncClient) -> None:
        resp = await client.post(
            "/api/v1/sessions/nonexistent/messages",
            json={"role": "user", "content": "test"},
        )
        assert resp.status_code == 404

    async def test_get_messages_api(self, client: AsyncClient) -> None:
        create = await client.post("/api/v1/sessions", json={})
        sid = create.json()["id"]
        await client.post(
            f"/api/v1/sessions/{sid}/messages",
            json={"role": "user", "content": "Hi"},
        )
        await client.post(
            f"/api/v1/sessions/{sid}/messages", json={"role": "assistant", "content": "Hello!"}
        )
        resp = await client.get(f"/api/v1/sessions/{sid}/messages")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["messages"]) == 2
        assert data["has_more"] is False
        assert data["next_cursor"] is None

    async def test_get_messages_session_not_found_api(self, client: AsyncClient) -> None:
        resp = await client.get("/api/v1/sessions/nonexistent/messages")
        assert resp.status_code == 404

    async def test_cursor_pagination_api(self, client: AsyncClient) -> None:
        create = await client.post("/api/v1/sessions", json={})
        sid = create.json()["id"]
        for i in range(5):
            await client.post(
                f"/api/v1/sessions/{sid}/messages",
                json={"role": "user", "content": f"msg-{i}"},
            )
        # Page 1
        resp = await client.get(f"/api/v1/sessions/{sid}/messages", params={"limit": 2})
        data = resp.json()
        assert len(data["messages"]) == 2
        assert data["has_more"] is True
        cursor = data["next_cursor"]
        assert cursor is not None

        # Page 2
        resp = await client.get(
            f"/api/v1/sessions/{sid}/messages", params={"limit": 2, "cursor": cursor}
        )
        data = resp.json()
        assert len(data["messages"]) == 2
        assert data["has_more"] is True

        # Page 3 (last)
        cursor = data["next_cursor"]
        resp = await client.get(
            f"/api/v1/sessions/{sid}/messages", params={"limit": 2, "cursor": cursor}
        )
        data = resp.json()
        assert len(data["messages"]) == 1
        assert data["has_more"] is False
        assert data["next_cursor"] is None

    async def test_auto_title_on_first_user_message(self, client: AsyncClient) -> None:
        create = await client.post("/api/v1/sessions", json={})
        sid = create.json()["id"]
        assert create.json()["title"] == ""

        await client.post(
            f"/api/v1/sessions/{sid}/messages",
            json={"role": "user", "content": "How do I use Python asyncio?"},
        )

        # Session should now have a title
        resp = await client.get(f"/api/v1/sessions/{sid}")
        assert resp.json()["title"] == "How do I use Python asyncio?"

    async def test_auto_title_only_on_first_message(self, client: AsyncClient) -> None:
        create = await client.post("/api/v1/sessions", json={})
        sid = create.json()["id"]

        # First message sets title
        await client.post(
            f"/api/v1/sessions/{sid}/messages",
            json={"role": "user", "content": "First question"},
        )
        # Second message should NOT change title
        await client.post(
            f"/api/v1/sessions/{sid}/messages",
            json={"role": "user", "content": "Second question"},
        )
        resp = await client.get(f"/api/v1/sessions/{sid}")
        assert resp.json()["title"] == "First question"

    async def test_message_metadata_preserved_api(self, client: AsyncClient) -> None:
        create = await client.post("/api/v1/sessions", json={})
        sid = create.json()["id"]
        resp = await client.post(
            f"/api/v1/sessions/{sid}/messages",
            json={"role": "assistant", "content": "Hi", "metadata": {"model": "gpt-4"}},
        )
        assert resp.json()["metadata"] == {"model": "gpt-4"}


# ---------------------------------------------------------------------------
# Database layer tests
# ---------------------------------------------------------------------------


class TestDatabase:
    async def test_get_db_context_manager(self) -> None:
        async with get_db() as db:
            assert db is not None

    async def test_wal_mode(self) -> None:
        async with get_db() as db:
            cursor = await db.execute("PRAGMA journal_mode")
            row = await cursor.fetchone()
            assert row is not None
            assert row[0] == "wal"

    async def test_foreign_keys_enabled(self) -> None:
        async with get_db() as db:
            cursor = await db.execute("PRAGMA foreign_keys")
            row = await cursor.fetchone()
            assert row is not None
            assert row[0] == 1

    async def test_migrations_table_exists(self) -> None:
        async with get_db() as db:
            cursor = await db.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='_migrations'"
            )
            row = await cursor.fetchone()
            assert row is not None

    async def test_sessions_table_exists(self) -> None:
        async with get_db() as db:
            cursor = await db.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='sessions'"
            )
            row = await cursor.fetchone()
            assert row is not None

    async def test_messages_table_exists(self) -> None:
        async with get_db() as db:
            cursor = await db.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='messages'"
            )
            row = await cursor.fetchone()
            assert row is not None

    async def test_migrations_idempotent(self) -> None:
        """Running get_db multiple times should not fail (migrations are idempotent)."""
        async with get_db() as _db1:
            pass
        async with get_db() as db2:
            cursor = await db2.execute("SELECT COUNT(*) FROM _migrations")
            row = await cursor.fetchone()
            assert row is not None
            assert row[0] >= 1

    async def test_session_to_dict(self) -> None:
        async with get_db() as db:
            mgr = SessionManager(db)
            session = await mgr.create_session(title="Test", channel="telegram")
            d = session.to_dict()
            assert d["id"] == session.id
            assert d["title"] == "Test"
            assert d["channel"] == "telegram"
            assert "created_at" in d
            assert "updated_at" in d
            assert d["message_count"] == 0
