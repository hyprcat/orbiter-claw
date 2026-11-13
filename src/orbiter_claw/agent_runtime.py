"""Agent runtime integration with orbiter-core Agent/Runner/Swarm.

Provides the AgentRuntime class that configures agents, handles model
fallback, and streams responses via the WebSocket gateway.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Any

from orbiter.agent import Agent
from orbiter.runner import run
from orbiter.swarm import Swarm
from orbiter.types import (
    ErrorEvent,
    Message,
    StreamEvent,
    TextEvent,
    ToolCallEvent,
    ToolResultEvent,
    UserMessage,
)

from orbiter_claw.gateway import Channel, Envelope, MessageType, manager

_log = logging.getLogger(__name__)

# Default system prompt when no SOUL.md is configured
_DEFAULT_INSTRUCTIONS = (
    "You are a helpful, concise personal AI assistant called Claw. "
    "You run locally on the user's machine for maximum privacy."
)


class AgentRuntime:
    """Manages the orbiter-core agent lifecycle and model configuration.

    Handles agent creation, model fallback, and streaming token delivery
    to WebSocket clients.
    """

    def __init__(
        self,
        *,
        model: str = "openai:gpt-4o",
        fallback_model: str | None = None,
        instructions: str = _DEFAULT_INSTRUCTIONS,
        provider: Any = None,
        fallback_provider: Any = None,
    ) -> None:
        self.model = model
        self.fallback_model = fallback_model
        self.instructions = instructions
        self._provider = provider
        self._fallback_provider = fallback_provider

    def create_agent(
        self,
        *,
        name: str = "claw",
        instructions: str | None = None,
        tools: list[Any] | None = None,
        max_steps: int = 10,
    ) -> Agent:
        """Create an orbiter-core Agent with the configured model."""
        return Agent(
            name=name,
            model=self.model,
            instructions=instructions or self.instructions,
            tools=tools,
            max_steps=max_steps,
        )

    def create_swarm(
        self,
        *,
        agents: list[Agent],
        flow: str | None = None,
        mode: str = "workflow",
    ) -> Swarm:
        """Create a Swarm for multi-step task orchestration."""
        return Swarm(agents=agents, flow=flow, mode=mode)

    async def run_agent(
        self,
        agent: Agent,
        input: str,
        *,
        messages: list[Message] | None = None,
        provider: Any = None,
    ) -> str:
        """Run the agent and return the final text output.

        Tries the primary model first, falls back to fallback_model on failure.
        """
        resolved_provider = provider or self._provider
        try:
            result = await run(agent, input, messages=messages, provider=resolved_provider)
            return result.output
        except Exception as exc:
            if self.fallback_model and self._fallback_provider:
                _log.warning(
                    "Primary model failed (%s), trying fallback: %s",
                    exc,
                    self.fallback_model,
                )
                fallback_agent = Agent(
                    name=agent.name,
                    model=self.fallback_model,
                    instructions=agent.instructions if isinstance(agent.instructions, str) else "",
                    tools=list(agent.tools.values()) if agent.tools else None,
                    max_steps=agent.max_steps,
                )
                result = await run(
                    fallback_agent, input, messages=messages, provider=self._fallback_provider
                )
                return result.output
            raise

    async def stream_to_websocket(
        self,
        agent: Agent,
        input: str,
        client_id: str,
        *,
        messages: list[Message] | None = None,
        provider: Any = None,
    ) -> str:
        """Stream agent execution to a WebSocket client via the gateway.

        Yields token events and tool_result events to the client's chat
        channel in real time. Returns the complete response text.
        """
        resolved_provider = provider or self._provider
        full_text: list[str] = []

        try:
            stream = self._stream_with_fallback(
                agent, input, messages=messages, provider=resolved_provider
            )
            async for event in stream:
                envelope = _event_to_envelope(event)
                if envelope is not None:
                    await manager.send_to_client(client_id, envelope)

                if isinstance(event, TextEvent):
                    full_text.append(event.text)

        except Exception as exc:
            error_envelope = Envelope(
                channel=Channel.CHAT,
                type=MessageType.ERROR,
                payload={"message": str(exc)},
            )
            await manager.send_to_client(client_id, error_envelope)
            raise

        return "".join(full_text)

    async def _stream_with_fallback(
        self,
        agent: Agent,
        input: str,
        *,
        messages: list[Message] | None = None,
        provider: Any = None,
    ) -> AsyncIterator[StreamEvent]:
        """Stream with fallback to secondary model on failure."""
        try:
            async for event in run.stream(  # pyright: ignore[reportFunctionMemberAccess]
                agent, input, messages=messages, provider=provider, detailed=True
            ):
                yield event
        except Exception as exc:
            if self.fallback_model and self._fallback_provider:
                _log.warning(
                    "Primary stream failed (%s), trying fallback: %s",
                    exc,
                    self.fallback_model,
                )
                fallback_agent = Agent(
                    name=agent.name,
                    model=self.fallback_model,
                    instructions=agent.instructions if isinstance(agent.instructions, str) else "",
                    tools=list(agent.tools.values()) if agent.tools else None,
                    max_steps=agent.max_steps,
                )
                async for event in run.stream(  # pyright: ignore[reportFunctionMemberAccess]
                    fallback_agent,
                    input,
                    messages=messages,
                    provider=self._fallback_provider,
                    detailed=True,
                ):
                    yield event
            else:
                raise


def _event_to_envelope(event: StreamEvent) -> Envelope | None:
    """Convert an orbiter-core StreamEvent to a WebSocket Envelope."""
    if isinstance(event, TextEvent):
        return Envelope(
            channel=Channel.CHAT,
            type=MessageType.TOKEN,
            payload={"text": event.text, "agent": event.agent_name},
        )
    if isinstance(event, ToolCallEvent):
        return Envelope(
            channel=Channel.CHAT,
            type=MessageType.TOOL_RESULT,
            payload={
                "tool_name": event.tool_name,
                "tool_call_id": event.tool_call_id,
                "status": "started",
                "agent": event.agent_name,
            },
        )
    if isinstance(event, ToolResultEvent):
        return Envelope(
            channel=Channel.CHAT,
            type=MessageType.TOOL_RESULT,
            payload={
                "tool_name": event.tool_name,
                "tool_call_id": event.tool_call_id,
                "result": event.result,
                "error": event.error,
                "success": event.success,
                "status": "completed",
                "agent": event.agent_name,
            },
        )
    if isinstance(event, ErrorEvent):
        return Envelope(
            channel=Channel.CHAT,
            type=MessageType.ERROR,
            payload={"message": event.error, "error_type": event.error_type},
        )
    return None


def messages_to_orbiter(messages: list[dict[str, Any]]) -> list[Message]:
    """Convert raw message dicts (from DB) to orbiter-core Message types."""
    from orbiter.types import AssistantMessage, SystemMessage, ToolResult

    result: list[Message] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "user":
            result.append(UserMessage(content=content))
        elif role == "assistant":
            result.append(AssistantMessage(content=content))
        elif role == "system":
            result.append(SystemMessage(content=content))
        elif role == "tool":
            result.append(
                ToolResult(
                    tool_call_id=msg.get("tool_call_id", ""),
                    tool_name=msg.get("tool_name", ""),
                    content=content,
                )
            )
    return result
