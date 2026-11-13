"""REST API routes for workspace memory file management."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from orbiter_claw.workspace import get_workspace

router = APIRouter(prefix="/workspace", tags=["workspace"])


class WorkspaceFileResponse(BaseModel):
    filename: str
    content: str


class UpdateWorkspaceFileRequest(BaseModel):
    content: str


@router.get("")
async def list_workspace_files() -> dict[str, object]:
    """List all workspace files with metadata."""
    mgr = get_workspace()
    return {"files": mgr.list_files()}


@router.get("/{filename}")
async def read_workspace_file(filename: str) -> dict[str, object]:
    """Read a specific workspace file."""
    mgr = get_workspace()
    content = mgr.read_file(filename)
    if content is None:
        raise HTTPException(status_code=404, detail=f"File not found: {filename}")
    return {"filename": filename, "content": content}


@router.put("/{filename}")
async def update_workspace_file(
    filename: str,
    body: UpdateWorkspaceFileRequest,
) -> dict[str, object]:
    """Update a workspace file. Creates the file if it doesn't exist."""
    if not filename.endswith(".md"):
        raise HTTPException(status_code=400, detail="Only .md files are supported")
    mgr = get_workspace()
    ok = mgr.write_file(filename, body.content)
    if not ok:
        raise HTTPException(status_code=400, detail=f"Failed to write file: {filename}")
    return {"filename": filename, "content": body.content}
