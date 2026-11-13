"""REST API routes at /api/v1/."""

from __future__ import annotations

from fastapi import APIRouter

from orbiter_claw.gateway import manager

router = APIRouter()


@router.get("/connections")
async def list_connections() -> dict[str, list[dict[str, object]]]:
    """List active WebSocket connections."""
    connections = []
    for client_id, conn in manager.active_connections.items():
        connections.append(
            {
                "client_id": client_id,
                "subscriptions": sorted(conn.subscriptions),
                "connected_at": conn.connected_at,
            }
        )
    return {"connections": connections}
