"""Tests for agent runtime integration with orbiter-core."""

from __future__ import annotations

from typing import Any

import pytest
from httpx import AsyncClient
from orbiter.tool import tool
from orbiter.types import (
    AssistantMessage,
    SystemMessage,
    TextEvent,
    ToolCall,
    ToolCallEvent,
    ToolResultEvent,
    Usage,
    UserMessage,
)
from pydantic import BaseModel

from orbiter_claw.agent_runtime import (
    AgentRuntime,
    _event_to_envelope,
    messages_to_orbiter,
)
from orbiter_claw.gateway import Channel, Envelope, MessageType, manager

# ---------------------------------------------------------------------------
# Mock provider for testing
# ---------------------------------------------------------------------------


class MockResponse(BaseModel):
    """Mimics the provider response object."""

    content: str = ""
    tool_calls: list[ToolCall] = []
    usage: Usage = Usage(input_tokens=10, output_tokens=20, total_tokens=30)


class MockStreamChunk(BaseModel):
    """Mimics a streaming chunk from a provider."""

    delta: str | None = None
    tool_call_deltas: list[Any] = []
    usage: Usage | None = None


class MockProvider:
    """Mock LLM provider with complete() and stream() methods."""

    def __init__(self, response_text: str = "Hello! I'm Claw.") -> None:
        self.response_text = response_text
        self.complete_calls: list[dict[str, Any]] = []
        self.stream_calls: list[dict[str, Any]] = []

    async def complete(
        self,
        messages: list[Any],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 1.0,
        max_tokens: int | None = None,
    ) -> MockResponse:
        self.complete_calls.append(
            {"messages": messages, "tools": tools, "temperature": temperature}
        )
        return MockResponse(content=self.response_text)

    async def stream(
        self,
        messages: list[Any],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 1.0,
        max_tokens: int | None = None,
    ) -> Any:
        self.stream_calls.append({"messages": messages, "tools": tools, "temperature": temperature})
        # Yield text chunks
        words = self.response_text.split(" ")
        for i, word in enumerate(words):
            chunk = word if i == 0 else f" {word}"
            yield MockStreamChunk(delta=chunk)
        # Final chunk with usage
        yield MockStreamChunk(usage=Usage(input_tokens=10, output_tokens=20, total_tokens=30))


class FailingProvider:
    """Mock provider that always raises an error."""

    async def complete(self, messages: list[Any], **kwargs: Any) -> MockResponse:
        raise RuntimeError("Provider unavailable")

    async def stream(self, messages: list[Any], **kwargs: Any) -> Any:
        raise RuntimeError("Provider unavailable")
        yield  # make it an async generator  # noqa: E501


# ---------------------------------------------------------------------------
# AgentRuntime tests
# ---------------------------------------------------------------------------


class TestAgentRuntime:
    def test_create_agent(self) -> None:
        """Test creating an agent with default settings."""
        rt = AgentRuntime(model="openai:gpt-4o")
        agent = rt.create_agent()
        assert agent.name == "claw"
        assert agent.model == "openai:gpt-4o"
        assert agent.max_steps == 10

    def test_create_agent_custom(self) -> None:
        """Test creating an agent with custom settings."""
        rt = AgentRuntime(model="anthropic:claude-3-opus")
        agent = rt.create_agent(
            name="custom-bot",
            instructions="Be helpful.",
            max_steps=5,
        )
        assert agent.name == "custom-bot"
        assert agent.model == "anthropic:claude-3-opus"
        assert agent.instructions == "Be helpful."
        assert agent.max_steps == 5

    def test_create_agent_with_tools(self) -> None:
        """Test creating an agent with tools."""

        @tool
        def greet(name: str) -> str:
            """Greet someone.

            Args:
                name: Name to greet.
            """
            return f"Hello, {name}!"

        rt = AgentRuntime()
        agent = rt.create_agent(tools=[greet])
        assert "greet" in agent.tools
        schemas = agent.get_tool_schemas()
        assert len(schemas) == 1
        assert schemas[0]["function"]["name"] == "greet"

    def test_create_swarm(self) -> None:
        """Test creating a Swarm for multi-agent orchestration."""
        rt = AgentRuntime()
        agent_a = rt.create_agent(name="agent_a")
        agent_b = rt.create_agent(name="agent_b")
        swarm = rt.create_swarm(agents=[agent_a, agent_b], flow="agent_a >> agent_b")
        assert swarm.flow_order == ["agent_a", "agent_b"]
        assert swarm.mode == "workflow"

    async def test_run_agent(self) -> None:
        """Test running an agent with a mock provider."""
        provider = MockProvider(response_text="The answer is 42.")
        rt = AgentRuntime(provider=provider)
        agent = rt.create_agent()
        result = await rt.run_agent(agent, "What is the answer?")
        assert result == "The answer is 42."
        assert len(provider.complete_calls) == 1

    async def test_run_agent_fallback(self) -> None:
        """Test model fallback when primary fails."""
        primary = FailingProvider()
        fallback = MockProvider(response_text="Fallback response.")
        rt = AgentRuntime(
            model="openai:gpt-4o",
            fallback_model="anthropic:claude-3-haiku",
            provider=primary,
            fallback_provider=fallback,
        )
        agent = rt.create_agent()
        result = await rt.run_agent(agent, "Hello")
        assert result == "Fallback response."
        assert len(fallback.complete_calls) == 1

    async def test_run_agent_no_fallback_raises(self) -> None:
        """Test that errors propagate when no fallback is configured."""
        primary = FailingProvider()
        rt = AgentRuntime(provider=primary)
        agent = rt.create_agent()
        with pytest.raises(Exception, match="Provider unavailable"):
            await rt.run_agent(agent, "Hello")

    async def test_model_selection_format(self) -> None:
        """Test provider:model format string is correctly parsed."""
        rt = AgentRuntime(model="anthropic:claude-3-opus")
        agent = rt.create_agent()
        assert agent.provider_name == "anthropic"
        assert agent.model_name == "claude-3-opus"

    async def test_model_selection_default_provider(self) -> None:
        """Test model without provider defaults to openai."""
        rt = AgentRuntime(model="gpt-4o")
        agent = rt.create_agent()
        assert agent.provider_name == "openai"
        assert agent.model_name == "gpt-4o"


# ---------------------------------------------------------------------------
# Streaming to WebSocket tests
# ---------------------------------------------------------------------------


class TestStreamToWebSocket:
    async def test_stream_tokens(self) -> None:
        """Test streaming token events to a WebSocket client."""
        provider = MockProvider(response_text="Hello world")
        rt = AgentRuntime(provider=provider)
        agent = rt.create_agent()

        # Track messages sent to client
        sent: list[Envelope] = []
        original_send = manager.send_to_client

        async def capture_send(client_id: str, envelope: Envelope) -> None:
            sent.append(envelope)

        manager.send_to_client = capture_send  # type: ignore[assignment]
        try:
            result = await rt.stream_to_websocket(agent, "Hi", client_id="test-123")
            assert result == "Hello world"
            # Should have token events
            token_envs = [e for e in sent if e.type == MessageType.TOKEN]
            assert len(token_envs) >= 1
            # All on chat channel
            assert all(e.channel == Channel.CHAT for e in token_envs)
            # Reconstruct text from tokens
            text = "".join(e.payload["text"] for e in token_envs)
            assert text == "Hello world"
        finally:
            manager.send_to_client = original_send  # type: ignore[assignment]

    async def test_stream_error_sends_error_envelope(self) -> None:
        """Test that errors during streaming send error envelope to client."""
        provider = FailingProvider()
        rt = AgentRuntime(provider=provider)
        agent = rt.create_agent()

        sent: list[Envelope] = []

        async def capture_send(client_id: str, envelope: Envelope) -> None:
            sent.append(envelope)

        original_send = manager.send_to_client
        manager.send_to_client = capture_send  # type: ignore[assignment]
        try:
            with pytest.raises(RuntimeError):
                await rt.stream_to_websocket(agent, "Hello", client_id="test-err")
            error_envs = [e for e in sent if e.type == MessageType.ERROR]
            assert len(error_envs) >= 1
            assert "unavailable" in error_envs[0].payload["message"].lower()
        finally:
            manager.send_to_client = original_send  # type: ignore[assignment]

    async def test_stream_fallback(self) -> None:
        """Test streaming falls back to secondary model on failure."""
        primary = FailingProvider()
        fallback = MockProvider(response_text="Fallback stream")
        rt = AgentRuntime(
            model="openai:gpt-4o",
            fallback_model="openai:gpt-3.5-turbo",
            provider=primary,
            fallback_provider=fallback,
        )
        agent = rt.create_agent()

        sent: list[Envelope] = []

        async def capture_send(client_id: str, envelope: Envelope) -> None:
            sent.append(envelope)

        original_send = manager.send_to_client
        manager.send_to_client = capture_send  # type: ignore[assignment]
        try:
            result = await rt.stream_to_websocket(agent, "Hi", client_id="test-fb")
            assert result == "Fallback stream"
            token_envs = [e for e in sent if e.type == MessageType.TOKEN]
            assert len(token_envs) >= 1
        finally:
            manager.send_to_client = original_send  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Event to Envelope conversion tests
# ---------------------------------------------------------------------------


class TestEventToEnvelope:
    def test_text_event(self) -> None:
        """Test TextEvent converts to TOKEN envelope."""
        event = TextEvent(text="Hello", agent_name="claw")
        env = _event_to_envelope(event)
        assert env is not None
        assert env.channel == Channel.CHAT
        assert env.type == MessageType.TOKEN
        assert env.payload["text"] == "Hello"
        assert env.payload["agent"] == "claw"

    def test_tool_call_event(self) -> None:
        """Test ToolCallEvent converts to TOOL_RESULT started envelope."""
        event = ToolCallEvent(
            tool_name="search",
            tool_call_id="tc-1",
            agent_name="claw",
        )
        env = _event_to_envelope(event)
        assert env is not None
        assert env.type == MessageType.TOOL_RESULT
        assert env.payload["tool_name"] == "search"
        assert env.payload["status"] == "started"

    def test_tool_result_event(self) -> None:
        """Test ToolResultEvent converts to TOOL_RESULT completed envelope."""
        event = ToolResultEvent(
            tool_name="search",
            tool_call_id="tc-1",
            result="Found 5 results",
            success=True,
            agent_name="claw",
        )
        env = _event_to_envelope(event)
        assert env is not None
        assert env.type == MessageType.TOOL_RESULT
        assert env.payload["status"] == "completed"
        assert env.payload["result"] == "Found 5 results"
        assert env.payload["success"] is True

    def test_error_event(self) -> None:
        """Test ErrorEvent converts to ERROR envelope."""
        from orbiter.types import ErrorEvent

        event = ErrorEvent(
            error="Something went wrong",
            error_type="RuntimeError",
            agent_name="claw",
        )
        env = _event_to_envelope(event)
        assert env is not None
        assert env.type == MessageType.ERROR
        assert env.payload["message"] == "Something went wrong"

    def test_unknown_event_returns_none(self) -> None:
        """Test that unhandled event types return None."""
        from orbiter.types import UsageEvent

        event = UsageEvent(
            usage=Usage(input_tokens=1, output_tokens=2, total_tokens=3),
            agent_name="claw",
        )
        env = _event_to_envelope(event)
        assert env is None


# ---------------------------------------------------------------------------
# Message conversion tests
# ---------------------------------------------------------------------------


class TestMessageConversion:
    def test_user_message(self) -> None:
        """Test converting user message dict to orbiter type."""
        msgs = messages_to_orbiter([{"role": "user", "content": "Hello"}])
        assert len(msgs) == 1
        assert isinstance(msgs[0], UserMessage)
        assert msgs[0].content == "Hello"

    def test_assistant_message(self) -> None:
        """Test converting assistant message dict to orbiter type."""
        msgs = messages_to_orbiter([{"role": "assistant", "content": "Hi there!"}])
        assert len(msgs) == 1
        assert isinstance(msgs[0], AssistantMessage)
        assert msgs[0].content == "Hi there!"

    def test_system_message(self) -> None:
        """Test converting system message dict to orbiter type."""
        msgs = messages_to_orbiter([{"role": "system", "content": "Be helpful."}])
        assert len(msgs) == 1
        assert isinstance(msgs[0], SystemMessage)
        assert msgs[0].content == "Be helpful."

    def test_tool_message(self) -> None:
        """Test converting tool message dict to orbiter type."""
        from orbiter.types import ToolResult

        msgs = messages_to_orbiter(
            [
                {
                    "role": "tool",
                    "content": "42",
                    "tool_call_id": "tc-1",
                    "tool_name": "calc",
                }
            ]
        )
        assert len(msgs) == 1
        assert isinstance(msgs[0], ToolResult)
        assert msgs[0].content == "42"
        assert msgs[0].tool_call_id == "tc-1"

    def test_mixed_messages(self) -> None:
        """Test converting a conversation with multiple message types."""
        msgs = messages_to_orbiter(
            [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello!"},
                {"role": "user", "content": "What is 2+2?"},
            ]
        )
        assert len(msgs) == 4
        assert isinstance(msgs[0], SystemMessage)
        assert isinstance(msgs[1], UserMessage)
        assert isinstance(msgs[2], AssistantMessage)
        assert isinstance(msgs[3], UserMessage)

    def test_empty_messages(self) -> None:
        """Test converting empty message list."""
        msgs = messages_to_orbiter([])
        assert msgs == []


# ---------------------------------------------------------------------------
# API endpoint tests
# ---------------------------------------------------------------------------


class TestCompletionsAPI:
    async def test_completions_endpoint(self, app: Any, client: AsyncClient) -> None:
        """Test POST /api/v1/completions returns agent response."""
        from orbiter_claw.routes import completions

        provider = MockProvider(response_text="API response.")
        original_runtime = completions.runtime
        completions.runtime = AgentRuntime(provider=provider)
        try:
            resp = await client.post(
                "/api/v1/completions",
                json={"message": "Hello", "model": "openai:gpt-4o"},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["response"] == "API response."
            assert data["model"] == "openai:gpt-4o"
        finally:
            completions.runtime = original_runtime

    async def test_completions_with_session_id(self, app: Any, client: AsyncClient) -> None:
        """Test completion request with session_id passes through."""
        from orbiter_claw.routes import completions

        provider = MockProvider(response_text="Session response.")
        original_runtime = completions.runtime
        completions.runtime = AgentRuntime(provider=provider)
        try:
            resp = await client.post(
                "/api/v1/completions",
                json={"message": "Hi", "session_id": "sess-123"},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["session_id"] == "sess-123"
        finally:
            completions.runtime = original_runtime

    async def test_completions_error(self, app: Any, client: AsyncClient) -> None:
        """Test completion endpoint returns 502 on agent failure."""
        from orbiter_claw.routes import completions

        original_runtime = completions.runtime
        completions.runtime = AgentRuntime(provider=FailingProvider())
        try:
            resp = await client.post(
                "/api/v1/completions",
                json={"message": "Hello"},
            )
            assert resp.status_code == 502
        finally:
            completions.runtime = original_runtime

    async def test_models_endpoint(self, app: Any, client: AsyncClient) -> None:
        """Test GET /api/v1/completions/models returns model config."""
        from orbiter_claw.routes import completions

        original_runtime = completions.runtime
        completions.runtime = AgentRuntime(
            model="openai:gpt-4o",
            fallback_model="anthropic:claude-3-haiku",
        )
        try:
            resp = await client.get("/api/v1/completions/models")
            assert resp.status_code == 200
            data = resp.json()
            assert data["primary"] == "openai:gpt-4o"
            assert data["fallback"] == "anthropic:claude-3-haiku"
        finally:
            completions.runtime = original_runtime

    async def test_completions_custom_system_prompt(self, app: Any, client: AsyncClient) -> None:
        """Test completion with custom system_prompt."""
        from orbiter_claw.routes import completions

        provider = MockProvider(response_text="Custom prompt response.")
        original_runtime = completions.runtime
        completions.runtime = AgentRuntime(provider=provider)
        try:
            resp = await client.post(
                "/api/v1/completions",
                json={
                    "message": "Hello",
                    "system_prompt": "Be a pirate.",
                },
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["response"] == "Custom prompt response."
        finally:
            completions.runtime = original_runtime


# ---------------------------------------------------------------------------
# Agent tool execution flow tests
# ---------------------------------------------------------------------------


class TestAgentToolFlow:
    async def test_agent_with_tool_calls(self) -> None:
        """Test that agent properly invokes tools and returns final text."""

        @tool
        def add(a: int, b: int) -> str:
            """Add two numbers.

            Args:
                a: First number.
                b: Second number.
            """
            return str(a + b)

        # Provider that returns a tool call first, then text
        call_count = 0

        class ToolProvider:
            async def complete(self, messages: list[Any], **kwargs: Any) -> MockResponse:
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    return MockResponse(
                        content="",
                        tool_calls=[
                            ToolCall(
                                id="tc-1",
                                name="add",
                                arguments='{"a": 2, "b": 3}',
                            )
                        ],
                    )
                return MockResponse(content="The sum is 5.")

        provider = ToolProvider()
        rt = AgentRuntime(provider=provider)
        agent = rt.create_agent(tools=[add])
        result = await rt.run_agent(agent, "What is 2+3?")
        assert result == "The sum is 5."
        assert call_count == 2
