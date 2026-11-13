"""Channel registry: manages adapter lifecycle, health monitoring, and plugin discovery."""

from __future__ import annotations

import asyncio
import contextlib
import importlib
import logging
import pkgutil
from importlib.metadata import entry_points
from typing import Any

from orbiter_claw.channels.adapter import (
    ChannelAdapter,
    ChannelStatus,
    MessageCallback,
)

logger = logging.getLogger(__name__)

# Health check interval in seconds
HEALTH_CHECK_INTERVAL = 30.0

# Exponential backoff for reconnection
RECONNECT_BASE_DELAY = 1.0
RECONNECT_MAX_DELAY = 60.0
RECONNECT_MULTIPLIER = 2.0


class ChannelRegistry:
    """Manages channel adapter registration, lifecycle, and health monitoring.

    Adapters are registered by name and can be started/stopped individually
    or all at once. Health monitoring runs periodic checks and auto-reconnects
    failed channels with exponential backoff.
    """

    def __init__(self) -> None:
        self._adapters: dict[str, type[ChannelAdapter]] = {}
        self._instances: dict[str, ChannelAdapter] = {}
        self._configs: dict[str, dict[str, Any]] = {}
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._health_task: asyncio.Task[None] | None = None
        self._reconnect_delays: dict[str, float] = {}
        self._on_message: MessageCallback | None = None
        self._stopping = False

    def register(
        self,
        name: str,
        adapter_class: type[ChannelAdapter],
        config: dict[str, Any] | None = None,
    ) -> None:
        """Register a channel adapter class by name."""
        self._adapters[name] = adapter_class
        if config:
            self._configs[name] = config

    def get(self, name: str) -> ChannelAdapter | None:
        """Get a running channel adapter instance by name."""
        return self._instances.get(name)

    def list_registered(self) -> list[str]:
        """List all registered adapter names."""
        return list(self._adapters.keys())

    def list_active(self) -> list[str]:
        """List names of currently active (instantiated) adapters."""
        return list(self._instances.keys())

    def set_message_callback(self, callback: MessageCallback) -> None:
        """Set the global message callback for all channels."""
        self._on_message = callback
        for instance in self._instances.values():
            instance.on_message(callback)

    async def start(self, name: str) -> None:
        """Start a single channel adapter by name."""
        if name in self._instances:
            logger.warning("Channel %s already running", name)
            return
        adapter_class = self._adapters.get(name)
        if adapter_class is None:
            raise KeyError(f"No adapter registered with name: {name}")

        config = self._configs.get(name, {})
        instance = adapter_class(config=config)
        instance.name = name
        if self._on_message:
            instance.on_message(self._on_message)

        instance.status = ChannelStatus.CONNECTING
        try:
            await instance.connect()
            instance.status = ChannelStatus.CONNECTED
        except Exception:
            instance.status = ChannelStatus.ERROR
            logger.exception("Failed to start channel %s", name)
            raise

        self._instances[name] = instance
        self._reconnect_delays.pop(name, None)
        logger.info("Channel %s started", name)

    async def stop(self, name: str) -> None:
        """Stop a single channel adapter by name."""
        instance = self._instances.pop(name, None)
        if instance is None:
            return
        try:
            await instance.disconnect()
            instance.status = ChannelStatus.DISCONNECTED
        except Exception:
            logger.exception("Error stopping channel %s", name)
        self._reconnect_delays.pop(name, None)
        logger.info("Channel %s stopped", name)

    async def start_all(self) -> None:
        """Start all registered channel adapters and begin health monitoring."""
        self._stopping = False
        for name in self._adapters:
            if name not in self._instances:
                with contextlib.suppress(Exception):
                    await self.start(name)
        self._start_health_monitor()

    async def stop_all(self) -> None:
        """Stop all running channel adapters and health monitoring."""
        self._stopping = True
        self._stop_health_monitor()
        names = list(self._instances.keys())
        for name in names:
            await self.stop(name)

    def _start_health_monitor(self) -> None:
        """Start periodic health check task."""
        if self._health_task is not None and not self._health_task.done():
            return
        self._health_task = asyncio.create_task(self._health_loop())

    def _stop_health_monitor(self) -> None:
        """Cancel health check task."""
        if self._health_task is not None:
            self._health_task.cancel()
            self._health_task = None

    async def _health_loop(self) -> None:
        """Periodically check health of all active channels."""
        while not self._stopping:
            try:
                await asyncio.sleep(HEALTH_CHECK_INTERVAL)
            except asyncio.CancelledError:
                return

            for name in list(self._instances.keys()):
                instance = self._instances.get(name)
                if instance is None:
                    continue
                try:
                    healthy = await instance.health_check()
                    if not healthy:
                        logger.warning("Channel %s health check failed", name)
                        await self._reconnect(name)
                except Exception:
                    logger.exception("Channel %s health check error", name)
                    await self._reconnect(name)

            # Also try to reconnect registered-but-not-running channels
            for name in self._adapters:
                if name not in self._instances and not self._stopping:
                    await self._reconnect(name)

    async def _reconnect(self, name: str) -> None:
        """Reconnect a channel with exponential backoff."""
        delay = self._reconnect_delays.get(name, RECONNECT_BASE_DELAY)

        # Remove existing instance if present
        old_instance = self._instances.pop(name, None)
        if old_instance:
            old_instance.status = ChannelStatus.RECONNECTING
            with contextlib.suppress(Exception):
                await old_instance.disconnect()

        logger.info("Reconnecting channel %s in %.1fs", name, delay)
        try:
            await asyncio.sleep(delay)
        except asyncio.CancelledError:
            return

        if self._stopping:
            return

        try:
            await self.start(name)
            self._reconnect_delays.pop(name, None)
        except Exception:
            # Increase backoff for next attempt
            next_delay = min(delay * RECONNECT_MULTIPLIER, RECONNECT_MAX_DELAY)
            self._reconnect_delays[name] = next_delay
            logger.warning("Channel %s reconnect failed, next attempt in %.1fs", name, next_delay)

    def discover_entry_points(self, group: str = "orbiter_claw.channels") -> int:
        """Discover and register channel adapters via Python entry_points.

        Entry points should map name -> ChannelAdapter subclass.
        Returns the number of newly discovered adapters.
        """
        count = 0
        eps = entry_points()
        # Python 3.12+ returns a SelectableGroups; 3.9-3.11 returns dict
        channel_eps = eps.select(group=group) if hasattr(eps, "select") else eps.get(group, [])
        for ep in channel_eps:
            if ep.name in self._adapters:
                continue
            try:
                adapter_class = ep.load()
                if isinstance(adapter_class, type) and issubclass(adapter_class, ChannelAdapter):
                    self.register(ep.name, adapter_class)
                    count += 1
                    logger.info("Discovered channel via entry_point: %s", ep.name)
                else:
                    logger.warning(
                        "Entry point %s does not point to a ChannelAdapter subclass", ep.name
                    )
            except Exception:
                logger.exception("Failed to load entry point %s", ep.name)
        return count

    def discover_package(self, package_name: str = "orbiter_claw.channels") -> int:
        """Discover channel adapters by scanning a Python package for modules.

        Each module should define a class named ``Adapter`` that subclasses
        ChannelAdapter. The module name (without the package prefix) is used
        as the channel name.

        Returns the number of newly discovered adapters.
        """
        count = 0
        try:
            package = importlib.import_module(package_name)
        except ImportError:
            logger.warning("Package %s not found for channel discovery", package_name)
            return 0

        if not hasattr(package, "__path__"):
            return 0

        for _importer, module_name, is_pkg in pkgutil.iter_modules(
            package.__path__, prefix=package_name + "."
        ):
            if is_pkg:
                continue
            # Skip internal modules
            short_name = module_name.rsplit(".", maxsplit=1)[-1]
            if short_name.startswith("_") or short_name in ("adapter", "registry"):
                continue
            if short_name in self._adapters:
                continue
            try:
                mod = importlib.import_module(module_name)
                adapter_class = getattr(mod, "Adapter", None)
                if (
                    adapter_class is not None
                    and isinstance(adapter_class, type)
                    and issubclass(adapter_class, ChannelAdapter)
                ):
                    self.register(short_name, adapter_class)
                    count += 1
                    logger.info("Discovered channel via package scan: %s", short_name)
            except Exception:
                logger.exception("Failed to load channel module %s", module_name)
        return count


def load_channel_configs(config_path: str | None = None) -> dict[str, dict[str, Any]]:
    """Load channel configurations from a TOML config file.

    Reads [channels.<name>] sections from the config file.
    Returns a dict mapping channel name to its config dict.
    """
    from pathlib import Path

    if config_path is None:
        config_path = str(Path.home() / ".orbiter-claw" / "config.toml")

    path = Path(config_path)
    if not path.exists():
        return {}

    try:
        import tomllib
    except ImportError:
        import tomli as tomllib  # type: ignore[no-redef]

    try:
        with open(path, "rb") as f:
            data = tomllib.load(f)
    except Exception:
        logger.exception("Failed to read config file %s", config_path)
        return {}

    channels_section = data.get("channels", {})
    if not isinstance(channels_section, dict):
        return {}

    result: dict[str, dict[str, Any]] = {}
    for name, cfg in channels_section.items():
        if isinstance(cfg, dict):
            result[name] = cfg
    return result
