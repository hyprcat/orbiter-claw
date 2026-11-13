"""Channel plugin system for orbiter-claw.

Provides abstract ChannelAdapter base class, ChannelRegistry for managing
channel lifecycles, and plugin discovery via entry_points and directory scanning.
"""

from orbiter_claw.channels.adapter import ChannelAdapter, ChannelMessage, ChannelStatus
from orbiter_claw.channels.registry import ChannelRegistry

__all__ = [
    "ChannelAdapter",
    "ChannelMessage",
    "ChannelRegistry",
    "ChannelStatus",
]
