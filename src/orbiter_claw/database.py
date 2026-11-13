"""Async SQLite database layer with WAL mode, foreign keys, and encryption support.

Migrations are stored as sequential .sql files in the migrations/ directory,
tracked in a _migrations table. Transparent encryption is provided via
EncryptionManager for message content and metadata fields.
"""

from __future__ import annotations

import os
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import aiosqlite

from orbiter_claw.encryption import EncryptionManager

_DEFAULT_DB_DIR = Path.home() / ".orbiter-claw"
_DB_FILENAME = "orbiter_claw.db"
_MIGRATIONS_DIR = Path(__file__).parent / "migrations"

# Module-level encryption manager — set via init_encryption()
_encryption: EncryptionManager | None = None


def init_encryption(manager: EncryptionManager) -> None:
    """Initialize the module-level encryption manager."""
    global _encryption
    _encryption = manager


def get_encryption() -> EncryptionManager | None:
    """Get the current encryption manager (None if encryption disabled)."""
    return _encryption


def encrypt_value(plaintext: str) -> str:
    """Encrypt a value if encryption is enabled, otherwise return as-is."""
    if _encryption is None:
        return plaintext
    return _encryption.encrypt(plaintext)


def decrypt_value(ciphertext: str) -> str:
    """Decrypt a value if encryption is enabled, otherwise return as-is."""
    if _encryption is None:
        return ciphertext
    return _encryption.decrypt(ciphertext)


def _get_db_path() -> str:
    """Get database file path, respecting ORBITER_CLAW_DB_PATH env var."""
    env_path = os.environ.get("ORBITER_CLAW_DB_PATH")
    if env_path:
        return env_path
    _DEFAULT_DB_DIR.mkdir(parents=True, exist_ok=True)
    return str(_DEFAULT_DB_DIR / _DB_FILENAME)


def _discover_migrations(migrations_dir: Path | None = None) -> list[tuple[int, str, str]]:
    """Discover .sql migration files sorted by numeric prefix.

    Returns list of (id, filename, sql_content) tuples.
    """
    mdir = migrations_dir or _MIGRATIONS_DIR
    if not mdir.is_dir():
        return []

    migrations: list[tuple[int, str, str]] = []
    for path in sorted(mdir.glob("*.sql")):
        # Extract numeric prefix from filename like "001_sessions_messages.sql"
        name = path.stem
        parts = name.split("_", 1)
        try:
            migration_id = int(parts[0])
        except (ValueError, IndexError):
            continue
        sql = path.read_text(encoding="utf-8")
        migrations.append((migration_id, path.name, sql))

    return migrations


async def _run_migrations(
    db: aiosqlite.Connection,
    migrations_dir: Path | None = None,
) -> list[int]:
    """Run pending migrations from .sql files. Returns list of applied migration IDs."""
    # Ensure _migrations table exists first
    await db.execute(
        "CREATE TABLE IF NOT EXISTS _migrations"
        " (id INTEGER PRIMARY KEY, filename TEXT, applied_at REAL NOT NULL)"
    )
    await db.commit()

    cursor = await db.execute("SELECT id FROM _migrations ORDER BY id")
    applied = {row[0] for row in await cursor.fetchall()}

    migrations = _discover_migrations(migrations_dir)
    newly_applied: list[int] = []

    for migration_id, filename, sql in migrations:
        if migration_id in applied:
            continue
        await db.executescript(sql)
        await db.execute(
            "INSERT INTO _migrations (id, filename, applied_at) VALUES (?, ?, ?)",
            (migration_id, filename, time.time()),
        )
        await db.commit()
        newly_applied.append(migration_id)

    return newly_applied


@asynccontextmanager
async def get_db(
    db_path: str | None = None,
    migrations_dir: Path | None = None,
) -> AsyncIterator[aiosqlite.Connection]:
    """Async context manager for database connections.

    Enables WAL mode and foreign keys on each connection.
    Runs pending migrations on first connection.
    """
    path = db_path or _get_db_path()
    db = await aiosqlite.connect(path)
    try:
        db.row_factory = aiosqlite.Row
        await db.execute("PRAGMA journal_mode=WAL")
        await db.execute("PRAGMA foreign_keys=ON")
        await _run_migrations(db, migrations_dir)
        yield db
    finally:
        await db.close()
