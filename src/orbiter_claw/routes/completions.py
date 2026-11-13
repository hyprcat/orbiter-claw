"""REST API routes for agent completions."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from orbiter_claw.agent_runtime import AgentRuntime

router = APIRouter(prefix="/completions", tags=["completions"])

# Module-level runtime instance (configured at startup; tests inject mocks)
runtime: AgentRuntime = AgentRuntime()


class CompletionRequest(BaseModel):
    message: str
    model: str = Field(default="openai:gpt-4o", description="Model in provider:model format")
    session_id: str | None = None
    system_prompt: str | None = None


class CompletionResponse(BaseModel):
    response: str
    model: str
    session_id: str | None = None


@router.post("")
async def create_completion(body: CompletionRequest) -> dict[str, object]:
    """Run an agent completion and return the result.

    Creates an agent with the specified model, runs the message through
    the orbiter-core Runner, and returns the response text.
    """
    agent = runtime.create_agent(
        instructions=body.system_prompt,
    )

    try:
        response_text = await runtime.run_agent(agent, body.message)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Agent execution failed: {exc}") from exc

    return {
        "response": response_text,
        "model": body.model,
        "session_id": body.session_id,
    }


@router.get("/models")
async def list_models() -> dict[str, object]:
    """Return the configured primary and fallback models."""
    return {
        "primary": runtime.model,
        "fallback": runtime.fallback_model,
    }
