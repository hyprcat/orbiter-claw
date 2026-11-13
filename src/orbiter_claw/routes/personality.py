"""REST API routes for personality (SOUL.md) management."""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from orbiter_claw.personality import get_personality

router = APIRouter(prefix="/personality", tags=["personality"])


class PersonalityResponse(BaseModel):
    raw: str
    sections: dict[str, str]


class UpdatePersonalityRequest(BaseModel):
    content: str


@router.get("")
async def get_personality_endpoint() -> dict[str, object]:
    """Read the current personality (SOUL.md content and parsed sections)."""
    mgr = get_personality()
    return {
        "raw": mgr.raw_content,
        "sections": mgr.sections,
    }


@router.put("")
async def update_personality(body: UpdatePersonalityRequest) -> dict[str, object]:
    """Update SOUL.md content. Takes effect on next message without restart."""
    mgr = get_personality()
    sections = mgr.update(body.content)
    return {
        "raw": mgr.raw_content,
        "sections": sections,
    }
