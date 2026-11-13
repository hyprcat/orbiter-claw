"""Tests for the channel plugin system (US-007)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from orbiter_claw.app import create_app
from orbiter_claw.channels.adapter import (
    ChannelAdapter,
    ChannelMessage,
    ChannelStatus,
)
from orbiter_claw.channels.registry import (
    RECONNECT_BASE_DELAY,
    RECONNECT_MAX_DELAY,
    RECONNECT_MULTIPLIER,
    ChannelRegistry,
    load_channel_configs,
)

# ---------------------------------------------------------------------------
# Mock adapter implementations for testing
# ---------------------------------------------------------------------------


class MockAdapter(ChannelAdapter):
    """A simple mock adapter that tracks calls."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.connect_count = 0
        self.disconnect_count = 0
        self.sent_messages: list[ChannelMessage] = []
        self._healthy = True

    async def connect(self) -> None:
        self.connect_count += 1
        self.status = ChannelStatus.CONNECTED

    async def disconnect(self) -> None:
        self.disconnect_count += 1
        self.status = ChannelStatus.DISCONNECTED

    async def send(self, message: ChannelMessage) -> None:
        self.sent_messages.append(message)

    async def health_check(self) -> bool:
        return self._healthy


class FailingAdapter(ChannelAdapter):
    """An adapter that raises on connect."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.connect_attempts = 0

    async def connect(self) -> None:
        self.connect_attempts += 1
        raise ConnectionError("Simulated connection failure")

    async def disconnect(self) -> None:
        self.status = ChannelStatus.DISCONNECTED

    async def send(self, message: ChannelMessage) -> None:
        raise ConnectionError("Not connected")

    async def health_check(self) -> bool:
        return False


class EventuallyConnectsAdapter(ChannelAdapter):
    """Fails N times then succeeds."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.connect_attempts = 0
        self.fail_count = config.get("fail_count", 2) if config else 2

    async def connect(self) -> None:
        self.connect_attempts += 1
        if self.connect_attempts <= self.fail_count:
            raise ConnectionError(f"Attempt {self.connect_attempts} failed")
        self.status = ChannelStatus.CONNECTED

    async def disconnect(self) -> None:
        self.status = ChannelStatus.DISCONNECTED

    async def send(self, message: ChannelMessage) -> None:
        pass

    async def health_check(self) -> bool:
        return self.status == ChannelStatus.CONNECTED


# ---------------------------------------------------------------------------
# ChannelMessage model tests
# ---------------------------------------------------------------------------


class TestChannelMessage:
    def test_basic_message(self) -> None:
        msg = ChannelMessage(role="user", content="hello", channel="telegram")
        assert msg.role == "user"
        assert msg.content == "hello"
        assert msg.channel == "telegram"
        assert msg.metadata == {}
        assert msg.timestamp > 0

    def test_message_with_metadata(self) -> None:
        metadata = {"chat_id": "12345", "username": "testuser"}
        msg = ChannelMessage(role="user", content="hi", channel="discord", metadata=metadata)
        assert msg.metadata["chat_id"] == "12345"
        assert msg.metadata["username"] == "testuser"

    def test_message_serialization(self) -> None:
        msg = ChannelMessage(
            role="assistant", content="response", channel="slack", timestamp=1000.0
        )
        data = msg.model_dump()
        assert data["role"] == "assistant"
        assert data["content"] == "response"
        assert data["channel"] == "slack"
        assert data["timestamp"] == 1000.0

    def test_message_from_dict(self) -> None:
        data = {
            "role": "user",
            "content": "test",
            "channel": "whatsapp",
            "timestamp": 999.0,
        }
        msg = ChannelMessage.model_validate(data)
        assert msg.role == "user"
        assert msg.timestamp == 999.0

    def test_message_roles(self) -> None:
        for role in ("user", "assistant", "system", "tool"):
            msg = ChannelMessage(role=role, content="x", channel="test")
            assert msg.role == role


# ---------------------------------------------------------------------------
# ChannelStatus enum tests
# ---------------------------------------------------------------------------


class TestChannelStatus:
    def test_all_statuses(self) -> None:
        expected = {"disconnected", "connecting", "connected", "reconnecting", "error"}
        actual = {s.value for s in ChannelStatus}
        assert actual == expected

    def test_status_is_str(self) -> None:
        assert str(ChannelStatus.CONNECTED) == "connected"
        assert f"status={ChannelStatus.ERROR}" == "status=error"


# ---------------------------------------------------------------------------
# ChannelAdapter interface tests
# ---------------------------------------------------------------------------


class TestChannelAdapter:
    def test_initial_status(self) -> None:
        adapter = MockAdapter()
        assert adapter.status == ChannelStatus.DISCONNECTED

    def test_config_stored(self) -> None:
        config = {"bot_token": "secret", "mode": "polling"}
        adapter = MockAdapter(config=config)
        assert adapter.config == config

    def test_default_config_empty(self) -> None:
        adapter = MockAdapter()
        assert adapter.config == {}

    async def test_connect_updates_status(self) -> None:
        adapter = MockAdapter()
        await adapter.connect()
        assert adapter.status == ChannelStatus.CONNECTED
        assert adapter.connect_count == 1

    async def test_disconnect_updates_status(self) -> None:
        adapter = MockAdapter()
        await adapter.connect()
        await adapter.disconnect()
        assert adapter.status == ChannelStatus.DISCONNECTED
        assert adapter.disconnect_count == 1

    async def test_send_message(self) -> None:
        adapter = MockAdapter()
        msg = ChannelMessage(role="assistant", content="hello", channel="test")
        await adapter.send(msg)
        assert len(adapter.sent_messages) == 1
        assert adapter.sent_messages[0].content == "hello"

    async def test_on_message_callback(self) -> None:
        adapter = MockAdapter()
        received: list[ChannelMessage] = []

        async def callback(msg: ChannelMessage) -> None:
            received.append(msg)

        adapter.on_message(callback)
        assert adapter._on_message_callback is callback

        # Simulate adapter invoking callback
        msg = ChannelMessage(role="user", content="incoming", channel="test")
        assert adapter._on_message_callback is not None
        await adapter._on_message_callback(msg)
        assert len(received) == 1
        assert received[0].content == "incoming"

    async def test_health_check(self) -> None:
        adapter = MockAdapter()
        assert await adapter.health_check() is True
        adapter._healthy = False
        assert await adapter.health_check() is False

    def test_abstract_methods_required(self) -> None:
        """Verify ChannelAdapter cannot be instantiated without implementing abstract methods."""
        with pytest.raises(TypeError, match="abstract method"):

            class IncompleteAdapter(ChannelAdapter):
                pass

            IncompleteAdapter()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# ChannelRegistry tests
# ---------------------------------------------------------------------------


class TestChannelRegistry:
    def test_register_adapter(self) -> None:
        registry = ChannelRegistry()
        registry.register("mock", MockAdapter)
        assert "mock" in registry.list_registered()

    def test_register_with_config(self) -> None:
        registry = ChannelRegistry()
        config = {"token": "abc"}
        registry.register("mock", MockAdapter, config=config)
        assert registry._configs["mock"] == config

    def test_list_registered_empty(self) -> None:
        registry = ChannelRegistry()
        assert registry.list_registered() == []

    def test_list_active_empty(self) -> None:
        registry = ChannelRegistry()
        assert registry.list_active() == []

    def test_get_returns_none_when_not_started(self) -> None:
        registry = ChannelRegistry()
        registry.register("mock", MockAdapter)
        assert registry.get("mock") is None

    async def test_start_adapter(self) -> None:
        registry = ChannelRegistry()
        registry.register("mock", MockAdapter)
        await registry.start("mock")
        instance = registry.get("mock")
        assert instance is not None
        assert instance.status == ChannelStatus.CONNECTED
        assert "mock" in registry.list_active()
        await registry.stop_all()

    async def test_start_sets_name(self) -> None:
        registry = ChannelRegistry()
        registry.register("telegram", MockAdapter)
        await registry.start("telegram")
        instance = registry.get("telegram")
        assert instance is not None
        assert instance.name == "telegram"
        await registry.stop_all()

    async def test_start_passes_config(self) -> None:
        registry = ChannelRegistry()
        config = {"token": "xyz"}
        registry.register("mock", MockAdapter, config=config)
        await registry.start("mock")
        instance = registry.get("mock")
        assert instance is not None
        assert instance.config == config
        await registry.stop_all()

    async def test_start_unregistered_raises(self) -> None:
        registry = ChannelRegistry()
        with pytest.raises(KeyError, match="No adapter registered"):
            await registry.start("nonexistent")

    async def test_start_already_running_warns(self) -> None:
        registry = ChannelRegistry()
        registry.register("mock", MockAdapter)
        await registry.start("mock")
        # Starting again should not raise
        await registry.start("mock")
        await registry.stop_all()

    async def test_start_failing_adapter(self) -> None:
        registry = ChannelRegistry()
        registry.register("fail", FailingAdapter)
        with pytest.raises(ConnectionError):
            await registry.start("fail")
        assert registry.get("fail") is None

    async def test_stop_adapter(self) -> None:
        registry = ChannelRegistry()
        registry.register("mock", MockAdapter)
        await registry.start("mock")
        await registry.stop("mock")
        assert registry.get("mock") is None
        assert "mock" not in registry.list_active()

    async def test_stop_nonexistent_is_noop(self) -> None:
        registry = ChannelRegistry()
        await registry.stop("nonexistent")  # Should not raise

    async def test_start_all(self) -> None:
        registry = ChannelRegistry()
        registry.register("a", MockAdapter)
        registry.register("b", MockAdapter)
        await registry.start_all()
        assert sorted(registry.list_active()) == ["a", "b"]
        await registry.stop_all()

    async def test_start_all_skips_failures(self) -> None:
        registry = ChannelRegistry()
        registry.register("good", MockAdapter)
        registry.register("bad", FailingAdapter)
        await registry.start_all()
        assert "good" in registry.list_active()
        assert "bad" not in registry.list_active()
        await registry.stop_all()

    async def test_stop_all(self) -> None:
        registry = ChannelRegistry()
        registry.register("a", MockAdapter)
        registry.register("b", MockAdapter)
        await registry.start_all()
        await registry.stop_all()
        assert registry.list_active() == []

    async def test_set_message_callback(self) -> None:
        registry = ChannelRegistry()
        registry.register("mock", MockAdapter)
        await registry.start("mock")

        received: list[ChannelMessage] = []

        async def handler(msg: ChannelMessage) -> None:
            received.append(msg)

        registry.set_message_callback(handler)

        instance = registry.get("mock")
        assert instance is not None
        assert instance._on_message_callback is handler
        await registry.stop_all()

    async def test_message_callback_set_before_start(self) -> None:
        registry = ChannelRegistry()
        registry.register("mock", MockAdapter)

        received: list[ChannelMessage] = []

        async def handler(msg: ChannelMessage) -> None:
            received.append(msg)

        registry.set_message_callback(handler)
        await registry.start("mock")

        instance = registry.get("mock")
        assert instance is not None
        assert instance._on_message_callback is handler
        await registry.stop_all()


# ---------------------------------------------------------------------------
# Health monitoring & reconnection tests
# ---------------------------------------------------------------------------


class TestHealthMonitoring:
    async def test_reconnect_on_health_failure(self) -> None:
        """When health check returns False, registry should reconnect."""
        registry = ChannelRegistry()
        registry.register("flaky", MockAdapter)
        await registry.start("flaky")

        instance = registry.get("flaky")
        assert instance is not None
        # Make health check fail
        assert isinstance(instance, MockAdapter)
        instance._healthy = False

        # Trigger a single reconnect cycle
        await registry._reconnect("flaky")

        # Should have a new instance
        new_instance = registry.get("flaky")
        assert new_instance is not None
        assert new_instance is not instance
        assert new_instance.status == ChannelStatus.CONNECTED
        await registry.stop_all()

    async def test_reconnect_exponential_backoff(self) -> None:
        """Backoff delay should increase on repeated failures."""
        registry = ChannelRegistry()
        registry.register("fail", FailingAdapter)

        # First reconnect attempt
        await registry._reconnect("fail")
        assert registry._reconnect_delays["fail"] == RECONNECT_BASE_DELAY * RECONNECT_MULTIPLIER

        # Second attempt
        await registry._reconnect("fail")
        expected = RECONNECT_BASE_DELAY * RECONNECT_MULTIPLIER * RECONNECT_MULTIPLIER
        assert registry._reconnect_delays["fail"] == expected

    async def test_reconnect_max_delay(self) -> None:
        """Backoff should cap at RECONNECT_MAX_DELAY."""
        registry = ChannelRegistry()
        registry.register("fail", FailingAdapter)

        # Set delay near max
        registry._reconnect_delays["fail"] = RECONNECT_MAX_DELAY
        await registry._reconnect("fail")
        assert registry._reconnect_delays["fail"] <= RECONNECT_MAX_DELAY

    async def test_reconnect_resets_on_success(self) -> None:
        """Successful reconnect should clear the backoff delay."""
        registry = ChannelRegistry()
        registry.register("mock", MockAdapter)

        # Set a backoff delay
        registry._reconnect_delays["mock"] = 10.0
        await registry._reconnect("mock")

        # Delay should be cleared on success
        assert "mock" not in registry._reconnect_delays
        await registry.stop_all()

    async def test_health_monitor_starts_with_start_all(self) -> None:
        registry = ChannelRegistry()
        registry.register("mock", MockAdapter)
        await registry.start_all()
        assert registry._health_task is not None
        assert not registry._health_task.done()
        await registry.stop_all()

    async def test_health_monitor_stops_with_stop_all(self) -> None:
        registry = ChannelRegistry()
        registry.register("mock", MockAdapter)
        await registry.start_all()
        await registry.stop_all()
        assert registry._health_task is None or registry._health_task.cancelled()


# ---------------------------------------------------------------------------
# Plugin discovery tests
# ---------------------------------------------------------------------------


class TestPluginDiscovery:
    def test_discover_entry_points_no_plugins(self) -> None:
        registry = ChannelRegistry()
        count = registry.discover_entry_points(group="nonexistent.group")
        assert count == 0

    def test_discover_entry_points_with_mock(self) -> None:
        registry = ChannelRegistry()

        mock_ep = MagicMock()
        mock_ep.name = "test_channel"
        mock_ep.load.return_value = MockAdapter

        mock_eps = MagicMock()
        mock_eps.select.return_value = [mock_ep]

        with patch("orbiter_claw.channels.registry.entry_points", return_value=mock_eps):
            count = registry.discover_entry_points()

        assert count == 1
        assert "test_channel" in registry.list_registered()

    def test_discover_entry_points_skips_non_adapter(self) -> None:
        registry = ChannelRegistry()

        mock_ep = MagicMock()
        mock_ep.name = "bad_plugin"
        mock_ep.load.return_value = str  # Not a ChannelAdapter subclass

        mock_eps = MagicMock()
        mock_eps.select.return_value = [mock_ep]

        with patch("orbiter_claw.channels.registry.entry_points", return_value=mock_eps):
            count = registry.discover_entry_points()

        assert count == 0
        assert "bad_plugin" not in registry.list_registered()

    def test_discover_entry_points_skips_duplicate(self) -> None:
        registry = ChannelRegistry()
        registry.register("existing", MockAdapter)

        mock_ep = MagicMock()
        mock_ep.name = "existing"
        mock_ep.load.return_value = MockAdapter

        mock_eps = MagicMock()
        mock_eps.select.return_value = [mock_ep]

        with patch("orbiter_claw.channels.registry.entry_points", return_value=mock_eps):
            count = registry.discover_entry_points()

        assert count == 0  # Skipped because already registered

    def test_discover_entry_points_handles_load_error(self) -> None:
        registry = ChannelRegistry()

        mock_ep = MagicMock()
        mock_ep.name = "broken"
        mock_ep.load.side_effect = ImportError("module not found")

        mock_eps = MagicMock()
        mock_eps.select.return_value = [mock_ep]

        with patch("orbiter_claw.channels.registry.entry_points", return_value=mock_eps):
            count = registry.discover_entry_points()

        assert count == 0

    def test_discover_package_with_adapter_module(self, tmp_path: Path) -> None:
        """Test package scanning by creating a temp module with an Adapter class."""
        # Create a fake package
        pkg_dir = tmp_path / "fake_channels"
        pkg_dir.mkdir()
        (pkg_dir / "__init__.py").write_text("")
        (pkg_dir / "telegram.py").write_text(
            "from orbiter_claw.channels.adapter import "
            "ChannelAdapter, ChannelStatus\n"
            "\n"
            "class Adapter(ChannelAdapter):\n"
            "    async def connect(self):\n"
            "        self.status = ChannelStatus.CONNECTED\n"
            "    async def disconnect(self):\n"
            "        self.status = ChannelStatus.DISCONNECTED\n"
            "    async def send(self, message): pass\n"
            "    async def health_check(self): return True\n"
        )

        # Add tmp_path to sys.path so the package is importable
        sys.path.insert(0, str(tmp_path))
        try:
            registry = ChannelRegistry()
            count = registry.discover_package("fake_channels")
            assert count == 1
            assert "telegram" in registry.list_registered()
        finally:
            sys.path.remove(str(tmp_path))
            # Clean up imported modules
            for key in list(sys.modules.keys()):
                if key.startswith("fake_channels"):
                    del sys.modules[key]

    def test_discover_package_skips_internal_modules(self) -> None:
        """Modules starting with _ or named adapter/registry are skipped."""
        registry = ChannelRegistry()
        # Scanning our own package should not re-register adapter or registry
        count = registry.discover_package("orbiter_claw.channels")
        assert count == 0

    def test_discover_package_nonexistent(self) -> None:
        registry = ChannelRegistry()
        count = registry.discover_package("nonexistent.package")
        assert count == 0


# ---------------------------------------------------------------------------
# Config loading tests
# ---------------------------------------------------------------------------


class TestConfigLoading:
    def test_load_nonexistent_file(self) -> None:
        configs = load_channel_configs("/tmp/nonexistent_config.toml")
        assert configs == {}

    def test_load_empty_channels(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config.toml"
        config_file.write_text("[general]\nname = 'test'\n")
        configs = load_channel_configs(str(config_file))
        assert configs == {}

    def test_load_channel_configs(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config.toml"
        config_file.write_text(
            "[channels.telegram]\n"
            'bot_token = "abc123"\n'
            'mode = "polling"\n'
            "\n"
            "[channels.discord]\n"
            'bot_token = "xyz789"\n'
        )
        configs = load_channel_configs(str(config_file))
        assert "telegram" in configs
        assert configs["telegram"]["bot_token"] == "abc123"
        assert configs["telegram"]["mode"] == "polling"
        assert "discord" in configs
        assert configs["discord"]["bot_token"] == "xyz789"

    def test_load_default_path(self) -> None:
        # Default path is ~/.orbiter-claw/config.toml; it may or may not exist
        configs = load_channel_configs()
        assert isinstance(configs, dict)

    def test_load_invalid_toml(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config.toml"
        config_file.write_text("this is not valid toml [[[")
        configs = load_channel_configs(str(config_file))
        assert configs == {}

    def test_load_channels_non_dict_value(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config.toml"
        config_file.write_text('[channels]\ntelegram = "not a dict"\n')
        configs = load_channel_configs(str(config_file))
        assert "telegram" not in configs


# ---------------------------------------------------------------------------
# Message normalization tests
# ---------------------------------------------------------------------------


class TestMessageNormalization:
    def test_telegram_style_message(self) -> None:
        """Messages from different channels use the same ChannelMessage format."""
        telegram_msg = ChannelMessage(
            role="user",
            content="Hello from Telegram",
            channel="telegram",
            metadata={"chat_id": "123", "message_id": "456"},
        )
        discord_msg = ChannelMessage(
            role="user",
            content="Hello from Discord",
            channel="discord",
            metadata={"guild_id": "g1", "channel_id": "c1"},
        )
        assert telegram_msg.channel == "telegram"
        assert discord_msg.channel == "discord"
        # Both have the same structure
        assert set(ChannelMessage.model_fields.keys()) == {
            "role",
            "content",
            "channel",
            "metadata",
            "timestamp",
        }

    def test_assistant_response_normalization(self) -> None:
        msg = ChannelMessage(
            role="assistant",
            content="Here's your answer",
            channel="slack",
            metadata={"thread_ts": "1234.5678"},
        )
        assert msg.role == "assistant"
        assert msg.channel == "slack"

    def test_system_message(self) -> None:
        msg = ChannelMessage(
            role="system",
            content="Channel connected",
            channel="webchat",
        )
        assert msg.role == "system"

    def test_tool_message(self) -> None:
        msg = ChannelMessage(
            role="tool",
            content='{"result": "file.txt created"}',
            channel="telegram",
            metadata={"tool_name": "filesystem"},
        )
        assert msg.role == "tool"
        assert msg.metadata["tool_name"] == "filesystem"


# ---------------------------------------------------------------------------
# API endpoint tests
# ---------------------------------------------------------------------------


class TestChannelsAPI:
    async def test_list_channels_empty(self) -> None:
        app = create_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/v1/channels")
            assert resp.status_code == 200
            assert resp.json() == []

    async def test_list_channels_with_registry(self) -> None:
        from orbiter_claw.routes.channels import init_channels_router

        registry = ChannelRegistry()
        registry.register("mock_a", MockAdapter)
        registry.register("mock_b", MockAdapter)
        await registry.start("mock_a")

        init_channels_router(registry)
        try:
            app = create_app()
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get("/api/v1/channels")
                assert resp.status_code == 200
                data = resp.json()
                names = {ch["name"] for ch in data}
                assert names == {"mock_a", "mock_b"}
                # mock_a is started, mock_b is not
                for ch in data:
                    if ch["name"] == "mock_a":
                        assert ch["status"] == "connected"
                    else:
                        assert ch["status"] == "not_started"
        finally:
            init_channels_router(None)  # type: ignore[arg-type]
            await registry.stop_all()


# ---------------------------------------------------------------------------
# Integration / lifecycle tests
# ---------------------------------------------------------------------------


class TestIntegration:
    async def test_full_lifecycle(self) -> None:
        """Register → start → send → health_check → stop full cycle."""
        registry = ChannelRegistry()
        registry.register("test", MockAdapter)

        received: list[ChannelMessage] = []

        async def handler(msg: ChannelMessage) -> None:
            received.append(msg)

        registry.set_message_callback(handler)
        await registry.start("test")

        instance = registry.get("test")
        assert instance is not None
        assert isinstance(instance, MockAdapter)

        # Send a message
        msg = ChannelMessage(role="assistant", content="hi", channel="test")
        await instance.send(msg)
        assert len(instance.sent_messages) == 1

        # Health check
        assert await instance.health_check() is True

        # Stop
        await registry.stop("test")
        assert registry.get("test") is None
        assert instance.status == ChannelStatus.DISCONNECTED

    async def test_multiple_channels_independent(self) -> None:
        """Multiple channels operate independently."""
        registry = ChannelRegistry()
        registry.register("a", MockAdapter, config={"id": "a"})
        registry.register("b", MockAdapter, config={"id": "b"})

        await registry.start_all()

        inst_a = registry.get("a")
        inst_b = registry.get("b")
        assert inst_a is not None and inst_b is not None
        assert inst_a is not inst_b
        assert inst_a.config["id"] == "a"
        assert inst_b.config["id"] == "b"

        # Stopping one doesn't affect the other
        await registry.stop("a")
        assert registry.get("a") is None
        assert registry.get("b") is not None

        await registry.stop_all()

    async def test_adapter_receives_messages_via_callback(self) -> None:
        """Simulate an adapter receiving a message and invoking the callback."""
        registry = ChannelRegistry()
        registry.register("test", MockAdapter)

        received: list[ChannelMessage] = []

        async def handler(msg: ChannelMessage) -> None:
            received.append(msg)

        registry.set_message_callback(handler)
        await registry.start("test")

        instance = registry.get("test")
        assert instance is not None
        assert instance._on_message_callback is not None

        # Simulate incoming message
        incoming = ChannelMessage(role="user", content="hello", channel="test")
        await instance._on_message_callback(incoming)
        assert len(received) == 1
        assert received[0].content == "hello"

        await registry.stop_all()

    async def test_config_loaded_and_applied(self, tmp_path: Path) -> None:
        """Load configs from TOML and apply to registry."""
        config_file = tmp_path / "config.toml"
        config_file.write_text('[channels.telegram]\nbot_token = "test_token"\nmode = "polling"\n')

        configs = load_channel_configs(str(config_file))
        registry = ChannelRegistry()
        for name, cfg in configs.items():
            registry.register(name, MockAdapter, config=cfg)

        await registry.start("telegram")
        instance = registry.get("telegram")
        assert instance is not None
        assert instance.config["bot_token"] == "test_token"
        assert instance.config["mode"] == "polling"
        await registry.stop_all()
