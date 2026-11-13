"""Shared test fixtures."""

from __future__ import annotations

import os

import pytest
from httpx import ASGITransport, AsyncClient

from orbiter_claw.app import create_app
from orbiter_claw.gateway import manager


@pytest.fixture(autouse=True)
def _use_temp_db(tmp_path: object) -> None:
    """Use a temporary database for each test."""
    db_path = os.path.join(str(tmp_path), "test.db")
    os.environ["ORBITER_CLAW_DB_PATH"] = db_path
    yield  # type: ignore[misc]
    os.environ.pop("ORBITER_CLAW_DB_PATH", None)


@pytest.fixture
def app():
    return create_app()


@pytest.fixture
async def client(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.fixture(autouse=True)
async def _clear_connections():
    """Reset connection manager between tests."""
    yield
    for client_id in list(manager.active_connections):
        await manager.disconnect(client_id)
