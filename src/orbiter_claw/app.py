"""FastAPI application factory."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from orbiter_claw.gateway import websocket_endpoint
from orbiter_claw.routes.api import router as api_router
from orbiter_claw.routes.channels import router as channels_router
from orbiter_claw.routes.completions import router as completions_router
from orbiter_claw.routes.personality import router as personality_router
from orbiter_claw.routes.sessions import router as sessions_router
from orbiter_claw.routes.workspace import router as workspace_router


def create_app() -> FastAPI:
    app = FastAPI(title="orbiter-claw", version="0.1.0")

    # CORS for web frontend (localhost dev)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # WebSocket gateway
    app.websocket("/ws")(websocket_endpoint)

    # Health check
    @app.get("/api/health")
    async def health_check() -> dict[str, str]:
        return {"status": "ok"}

    # REST API
    app.include_router(api_router, prefix="/api/v1")
    app.include_router(sessions_router, prefix="/api/v1")
    app.include_router(completions_router, prefix="/api/v1")
    app.include_router(personality_router, prefix="/api/v1")
    app.include_router(workspace_router, prefix="/api/v1")
    app.include_router(channels_router, prefix="/api/v1")

    return app
