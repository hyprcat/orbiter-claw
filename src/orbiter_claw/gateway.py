"""WebSocket gateway with multiplexed channels and connection management."""

from __future__ import annotations

import asyncio
import contextlib
import time
import uuid
from enum import StrEnum
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field


class Channel(StrEnum):
    CHAT = "chat"
    SYSTEM = "system"
    NOTIFICATIONS = "notifications"


class MessageType(StrEnum):
    MESSAGE = "message"
    TOKEN = "token"
    TOOL_RESULT = "tool_result"
    ERROR = "error"
    PING = "ping"
    PONG = "pong"
    SUBSCRIBE = "subscribe"
    UNSUBSCRIBE = "unsubscribe"
    ACK = "ack"


class Envelope(BaseModel):
    channel: Channel
    type: MessageType
    payload: dict[str, Any] = Field(default_factory=dict)
    timestamp: float = Field(default_factory=time.time)


class ClientConnection:
    """Tracks a single WebSocket client connection."""

    def __init__(self, websocket: WebSocket, client_id: str) -> None:
        self.websocket = websocket
        self.client_id = client_id
        self.subscriptions: set[Channel] = {Channel.CHAT, Channel.SYSTEM, Channel.NOTIFICATIONS}
        self.connected_at: float = time.time()
        self.last_pong: float = time.time()


class ConnectionManager:
    """Manages active WebSocket connections."""

    def __init__(self) -> None:
        self._connections: dict[str, ClientConnection] = {}
        self._lock = asyncio.Lock()

    @property
    def active_connections(self) -> dict[str, ClientConnection]:
        return dict(self._connections)

    async def connect(self, websocket: WebSocket) -> ClientConnection:
        await websocket.accept()
        client_id = str(uuid.uuid4())
        conn = ClientConnection(websocket, client_id)
        async with self._lock:
            self._connections[client_id] = conn
        return conn

    async def disconnect(self, client_id: str) -> None:
        async with self._lock:
            self._connections.pop(client_id, None)

    async def send_to_client(self, client_id: str, envelope: Envelope) -> None:
        conn = self._connections.get(client_id)
        if conn is None:
            return
        await conn.websocket.send_json(envelope.model_dump())

    async def broadcast(self, envelope: Envelope, exclude: str | None = None) -> None:
        async with self._lock:
            targets = list(self._connections.values())
        for conn in targets:
            if conn.client_id == exclude:
                continue
            if envelope.channel in conn.subscriptions:
                with contextlib.suppress(Exception):
                    await conn.websocket.send_json(envelope.model_dump())


manager = ConnectionManager()


async def _handle_control_message(conn: ClientConnection, envelope: Envelope) -> Envelope | None:
    """Handle subscribe/unsubscribe/ping control messages."""
    if envelope.type == MessageType.PING:
        conn.last_pong = time.time()
        return Envelope(channel=Channel.SYSTEM, type=MessageType.PONG, payload={})

    if envelope.type == MessageType.SUBSCRIBE:
        channel_name = envelope.payload.get("channel")
        try:
            channel = Channel(channel_name)
            conn.subscriptions.add(channel)
            return Envelope(
                channel=Channel.SYSTEM,
                type=MessageType.ACK,
                payload={"action": "subscribed", "channel": channel_name},
            )
        except ValueError:
            return Envelope(
                channel=Channel.SYSTEM,
                type=MessageType.ERROR,
                payload={"message": f"Unknown channel: {channel_name}"},
            )

    if envelope.type == MessageType.UNSUBSCRIBE:
        channel_name = envelope.payload.get("channel")
        try:
            channel = Channel(channel_name)
            conn.subscriptions.discard(channel)
            return Envelope(
                channel=Channel.SYSTEM,
                type=MessageType.ACK,
                payload={"action": "unsubscribed", "channel": channel_name},
            )
        except ValueError:
            return Envelope(
                channel=Channel.SYSTEM,
                type=MessageType.ERROR,
                payload={"message": f"Unknown channel: {channel_name}"},
            )

    return None


async def websocket_endpoint(websocket: WebSocket) -> None:
    """Main WebSocket endpoint handler with multiplexed channel routing."""
    conn = await manager.connect(websocket)

    # Send welcome with client_id
    welcome = Envelope(
        channel=Channel.SYSTEM,
        type=MessageType.ACK,
        payload={"client_id": conn.client_id, "message": "connected"},
    )
    await websocket.send_json(welcome.model_dump())

    try:
        while True:
            data = await websocket.receive_json()
            try:
                envelope = Envelope.model_validate(data)
            except Exception:
                error = Envelope(
                    channel=Channel.SYSTEM,
                    type=MessageType.ERROR,
                    payload={"message": "Invalid message format"},
                )
                await websocket.send_json(error.model_dump())
                continue

            # Handle control messages
            response = await _handle_control_message(conn, envelope)
            if response is not None:
                await websocket.send_json(response.model_dump())
                continue

            # Route message to channel subscribers (broadcast)
            await manager.broadcast(envelope, exclude=conn.client_id)

            # ACK the message back to sender
            ack = Envelope(
                channel=envelope.channel,
                type=MessageType.ACK,
                payload={"received": True},
            )
            await websocket.send_json(ack.model_dump())

    except WebSocketDisconnect:
        pass
    finally:
        await manager.disconnect(conn.client_id)
