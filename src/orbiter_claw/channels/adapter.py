"""Abstract ChannelAdapter base class and unified message model."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class ChannelStatus(StrEnum):
    """Lifecycle status of a channel adapter."""

    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    ERROR = "error"


class ChannelMessage(BaseModel):
    """Unified message format produced by all channel adapters.

    All channels normalize their incoming messages into this format before
    passing them to the core message handler.
    """

    role: str
    content: str
    channel: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    timestamp: float = Field(default_factory=time.time)


MessageCallback = Callable[[ChannelMessage], Awaitable[None]]


class ChannelAdapter(ABC):
    """Abstract base class for messaging channel adapters.

    Each channel (Telegram, Discord, Slack, etc.) implements this interface
    to provide a uniform way to connect, send, receive, and monitor health.
    """

    name: str = ""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config: dict[str, Any] = config or {}
        self.status: ChannelStatus = ChannelStatus.DISCONNECTED
        self._on_message_callback: MessageCallback | None = None

    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to the channel service.

        Should set self.status to CONNECTED on success.
        """

    @abstractmethod
    async def disconnect(self) -> None:
        """Gracefully disconnect from the channel service.

        Should set self.status to DISCONNECTED on completion.
        """

    @abstractmethod
    async def send(self, message: ChannelMessage) -> None:
        """Send a message through this channel."""

    def on_message(self, callback: MessageCallback) -> None:
        """Register a callback to receive incoming messages.

        The callback receives a ChannelMessage and is invoked by the adapter
        whenever a new message arrives from the channel.
        """
        self._on_message_callback = callback

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the channel connection is healthy.

        Returns True if the channel is operational, False otherwise.
        """
