"""Tests for database layer: migrations, encryption integration, get_db()."""

from __future__ import annotations

from pathlib import Path

import aiosqlite

from orbiter_claw.database import (
    _discover_migrations,
    _run_migrations,
    decrypt_value,
    encrypt_value,
    get_db,
    get_encryption,
    init_encryption,
)
from orbiter_claw.encryption import EncryptionManager, generate_key


class TestMigrationDiscovery:
    """Tests for .sql file-based migration discovery."""

    def test_discover_migrations_from_project_dir(self) -> None:
        """Should find the project's migration files."""
        migrations = _discover_migrations()
        assert len(migrations) >= 2
        ids = [m[0] for m in migrations]
        assert 1 in ids
        assert 2 in ids

    def test_discover_migrations_sorted_by_id(self) -> None:
        migrations = _discover_migrations()
        ids = [m[0] for m in migrations]
        assert ids == sorted(ids)

    def test_discover_migrations_from_custom_dir(self, tmp_path: Path) -> None:
        """Should discover migrations from a custom directory."""
        (tmp_path / "001_init.sql").write_text(
            "CREATE TABLE IF NOT EXISTS test1 (id TEXT PRIMARY KEY);"
        )
        (tmp_path / "002_add_col.sql").write_text(
            "CREATE TABLE IF NOT EXISTS test2 (id TEXT PRIMARY KEY);"
        )
        migrations = _discover_migrations(tmp_path)
        assert len(migrations) == 2
        assert migrations[0][0] == 1
        assert migrations[1][0] == 2

    def test_discover_ignores_non_sql_files(self, tmp_path: Path) -> None:
        (tmp_path / "001_init.sql").write_text("CREATE TABLE t (id TEXT);")
        (tmp_path / "readme.md").write_text("Not a migration")
        (tmp_path / "data.json").write_text("{}")
        migrations = _discover_migrations(tmp_path)
        assert len(migrations) == 1

    def test_discover_ignores_bad_prefixes(self, tmp_path: Path) -> None:
        (tmp_path / "abc_init.sql").write_text("CREATE TABLE t (id TEXT);")
        migrations = _discover_migrations(tmp_path)
        assert len(migrations) == 0

    def test_discover_empty_directory(self, tmp_path: Path) -> None:
        migrations = _discover_migrations(tmp_path)
        assert migrations == []

    def test_discover_nonexistent_directory(self) -> None:
        migrations = _discover_migrations(Path("/nonexistent"))
        assert migrations == []


class TestRunMigrations:
    """Tests for migration execution and tracking."""

    async def test_creates_migrations_table(self, tmp_path: Path) -> None:
        db_path = str(tmp_path / "test.db")
        async with aiosqlite.connect(db_path) as db:
            await _run_migrations(db, tmp_path)
            cursor = await db.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='_migrations'"
            )
            row = await cursor.fetchone()
            assert row is not None

    async def test_runs_pending_migrations(self, tmp_path: Path) -> None:
        mdir = tmp_path / "migrations"
        mdir.mkdir()
        (mdir / "001_init.sql").write_text(
            "CREATE TABLE IF NOT EXISTS users (id TEXT PRIMARY KEY, name TEXT);"
        )

        db_path = str(tmp_path / "test.db")
        async with aiosqlite.connect(db_path) as db:
            applied = await _run_migrations(db, mdir)
            assert applied == [1]
            # Verify table was created
            cursor = await db.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='users'"
            )
            row = await cursor.fetchone()
            assert row is not None

    async def test_skips_already_applied(self, tmp_path: Path) -> None:
        mdir = tmp_path / "migrations"
        mdir.mkdir()
        (mdir / "001_init.sql").write_text(
            "CREATE TABLE IF NOT EXISTS items (id TEXT PRIMARY KEY);"
        )

        db_path = str(tmp_path / "test.db")
        async with aiosqlite.connect(db_path) as db:
            first = await _run_migrations(db, mdir)
            assert first == [1]
            second = await _run_migrations(db, mdir)
            assert second == []

    async def test_runs_new_migrations_only(self, tmp_path: Path) -> None:
        mdir = tmp_path / "migrations"
        mdir.mkdir()
        (mdir / "001_init.sql").write_text("CREATE TABLE IF NOT EXISTS t1 (id TEXT PRIMARY KEY);")

        db_path = str(tmp_path / "test.db")
        async with aiosqlite.connect(db_path) as db:
            await _run_migrations(db, mdir)

        # Add a new migration
        (mdir / "002_extend.sql").write_text("CREATE TABLE IF NOT EXISTS t2 (id TEXT PRIMARY KEY);")

        async with aiosqlite.connect(db_path) as db:
            applied = await _run_migrations(db, mdir)
            assert applied == [2]

    async def test_records_migration_metadata(self, tmp_path: Path) -> None:
        mdir = tmp_path / "migrations"
        mdir.mkdir()
        (mdir / "001_first.sql").write_text("CREATE TABLE IF NOT EXISTS x (id TEXT);")

        db_path = str(tmp_path / "test.db")
        async with aiosqlite.connect(db_path) as db:
            db.row_factory = aiosqlite.Row
            await _run_migrations(db, mdir)
            cursor = await db.execute("SELECT id, filename FROM _migrations")
            row = await cursor.fetchone()
            assert row is not None
            assert row["id"] == 1
            assert row["filename"] == "001_first.sql"


class TestGetDb:
    """Tests for get_db() context manager."""

    async def test_get_db_sets_wal_mode(self) -> None:
        async with get_db() as db:
            cursor = await db.execute("PRAGMA journal_mode")
            row = await cursor.fetchone()
            assert row is not None
            assert row[0] == "wal"

    async def test_get_db_enables_foreign_keys(self) -> None:
        async with get_db() as db:
            cursor = await db.execute("PRAGMA foreign_keys")
            row = await cursor.fetchone()
            assert row is not None
            assert row[0] == 1

    async def test_get_db_runs_migrations(self) -> None:
        async with get_db() as db:
            # sessions table should exist from migration 001
            cursor = await db.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='sessions'"
            )
            row = await cursor.fetchone()
            assert row is not None

    async def test_get_db_custom_path(self, tmp_path: Path) -> None:
        db_path = str(tmp_path / "custom.db")
        async with get_db(db_path=db_path) as db:
            cursor = await db.execute("SELECT 1")
            row = await cursor.fetchone()
            assert row is not None
        assert (tmp_path / "custom.db").exists()

    async def test_get_db_closes_connection(self) -> None:
        async with get_db() as _db:
            pass
        # After context manager exits, connection should be closed
        # aiosqlite doesn't expose a closed attribute easily,
        # but we can verify it doesn't raise during cleanup


class TestTransparentEncryption:
    """Tests for encrypt_value/decrypt_value with module-level encryption."""

    def test_no_encryption_passthrough(self) -> None:
        """With no encryption manager, values pass through unchanged."""
        # Reset encryption state
        init_encryption(None)  # type: ignore[arg-type]
        assert encrypt_value("hello") == "hello"
        assert decrypt_value("hello") == "hello"

    def test_encrypt_decrypt_with_manager(self) -> None:
        key = generate_key()
        mgr = EncryptionManager(key)
        init_encryption(mgr)
        try:
            encrypted = encrypt_value("secret data")
            assert encrypted != "secret data"
            decrypted = decrypt_value(encrypted)
            assert decrypted == "secret data"
        finally:
            init_encryption(None)  # type: ignore[arg-type]

    def test_get_encryption_returns_manager(self) -> None:
        key = generate_key()
        mgr = EncryptionManager(key)
        init_encryption(mgr)
        try:
            assert get_encryption() is mgr
        finally:
            init_encryption(None)  # type: ignore[arg-type]

    def test_get_encryption_returns_none(self) -> None:
        init_encryption(None)  # type: ignore[arg-type]
        assert get_encryption() is None


class TestEncryptedSessions:
    """Tests that sessions module transparently encrypts/decrypts messages."""

    async def test_message_content_encrypted_at_rest(self) -> None:
        """Verify that raw DB content is encrypted when encryption is enabled."""
        from orbiter_claw.sessions import SessionManager

        key = generate_key()
        mgr = EncryptionManager(key)
        init_encryption(mgr)
        try:
            async with get_db() as db:
                sm = SessionManager(db)
                session = await sm.create_session(title="test")
                msg = await sm.add_message(session.id, "user", "secret message")

                # Read raw content from DB — should be encrypted
                cursor = await db.execute("SELECT content FROM messages WHERE id = ?", (msg.id,))
                row = await cursor.fetchone()
                assert row is not None
                raw_content = row[0]
                assert raw_content != "secret message"

                # But reading through SessionManager should decrypt
                result = await sm.get_messages(session.id)
                assert result.messages[0].content == "secret message"
        finally:
            init_encryption(None)  # type: ignore[arg-type]

    async def test_message_metadata_encrypted_at_rest(self) -> None:
        """Verify that raw DB metadata is encrypted when encryption is enabled."""
        from orbiter_claw.sessions import SessionManager

        key = generate_key()
        mgr = EncryptionManager(key)
        init_encryption(mgr)
        try:
            async with get_db() as db:
                sm = SessionManager(db)
                session = await sm.create_session(title="test")
                meta = {"tool": "calculator", "result": 42}
                msg = await sm.add_message(session.id, "tool", "result", meta)

                # Read raw metadata from DB — should be encrypted
                cursor = await db.execute("SELECT metadata FROM messages WHERE id = ?", (msg.id,))
                row = await cursor.fetchone()
                assert row is not None
                raw_metadata = row[0]
                assert "calculator" not in raw_metadata

                # But reading through SessionManager should decrypt
                result = await sm.get_messages(session.id)
                assert result.messages[0].metadata["tool"] == "calculator"
        finally:
            init_encryption(None)  # type: ignore[arg-type]

    async def test_sessions_work_without_encryption(self) -> None:
        """Verify sessions work normally when encryption is disabled."""
        from orbiter_claw.sessions import SessionManager

        init_encryption(None)  # type: ignore[arg-type]
        async with get_db() as db:
            sm = SessionManager(db)
            session = await sm.create_session(title="plain")
            await sm.add_message(session.id, "user", "plain message")
            result = await sm.get_messages(session.id)
            assert result.messages[0].content == "plain message"
