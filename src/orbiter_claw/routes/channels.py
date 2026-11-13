"""REST API endpoints for channel management."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from orbiter_claw.channels.registry import ChannelRegistry

router = APIRouter(tags=["channels"])

# Module-level registry; set by app initialization
_registry: ChannelRegistry | None = None


def init_channels_router(registry: ChannelRegistry) -> None:
    """Wire up the channels router with a registry instance."""
    global _registry  # noqa: PLW0603
    _registry = registry


@router.get("/channels")
async def list_channels() -> list[dict[str, Any]]:
    """List all registered channels and their current status."""
    if _registry is None:
        return []
    result: list[dict[str, Any]] = []
    for name in _registry.list_registered():
        instance = _registry.get(name)
        status = instance.status if instance else "not_started"
        result.append({"name": name, "status": str(status)})
    return result
