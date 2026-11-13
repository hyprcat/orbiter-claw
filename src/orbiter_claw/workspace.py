"""Workspace memory file system.

Manages structured workspace files (SOUL.md, USER.md, IDENTITY.md, AGENTS.md,
TOOLS.md, MEMORY.md) in ~/.orbiter-claw/workspace/. Provides hot-reload on
file change, CRUD API, and context injection into SystemPromptBuilder.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import TYPE_CHECKING

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer  # pyright: ignore[reportMissingTypeStubs]

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)

DEFAULT_WORKSPACE_DIR = Path.home() / ".orbiter-claw" / "workspace"

# Workspace files in priority order (matches WORKSPACE_FILES in prompt_builder)
WORKSPACE_FILE_NAMES = (
    "SOUL.md",
    "AGENTS.md",
    "USER.md",
    "IDENTITY.md",
    "TOOLS.md",
    "MEMORY.md",
)

DEFAULT_USER_MD = """\
# User Profile
<!-- Add your personal info so the assistant can personalize responses -->

## Name
<!-- Your preferred name -->

## Timezone
<!-- e.g., America/New_York, Europe/London, Asia/Tokyo -->

## Work Context
<!-- What you work on, your role, industry -->

## Communication Preferences
<!-- e.g., prefer concise answers, use bullet points, include examples -->
"""

DEFAULT_IDENTITY_MD = """\
# Identity
name: orbiter-claw
version: 0.1.0

## Capabilities
- Multi-LLM agent execution
- Persistent conversation memory
- Encrypted local storage
- WebSocket real-time streaming
- Multi-channel messaging
- Tool execution (MCP servers)
- Personality customization (SOUL.md)
- Workspace memory (USER.md, MEMORY.md, AGENTS.md, TOOLS.md)
"""

DEFAULT_AGENTS_MD = """\
# Operational Rules

## Memory Management
- Append important facts, preferences, and patterns to MEMORY.md
- Review MEMORY.md before answering questions about prior work
- Keep memory entries concise and timestamped

## Safety Rules
- Never share user data externally
- Prefer trash over rm (recoverable beats gone forever)
- Ask before performing destructive or external actions

## When to Speak vs Stay Quiet
- Respond to direct messages immediately
- On heartbeat: only speak if something genuinely needs attention
- Use <<SILENT>> when there is nothing to say
"""

DEFAULT_TOOLS_MD = """\
# Available Tools
<!-- Auto-generated from registered tools and MCP servers -->
<!-- This file is updated automatically when tools change -->

No tools currently registered.
"""

DEFAULT_MEMORY_MD = """\
# Memory
<!-- Long-term memory: the agent appends learned facts, preferences, and
     patterns here over time. You can also add notes manually. -->
"""

# Maps filename -> default content
DEFAULT_FILE_CONTENTS: dict[str, str] = {
    "USER.md": DEFAULT_USER_MD,
    "IDENTITY.md": DEFAULT_IDENTITY_MD,
    "AGENTS.md": DEFAULT_AGENTS_MD,
    "TOOLS.md": DEFAULT_TOOLS_MD,
    "MEMORY.md": DEFAULT_MEMORY_MD,
}


class _WorkspaceFileHandler(FileSystemEventHandler):
    """Watches for workspace file changes and triggers a callback."""

    def __init__(self, callback: Callable[[], None]) -> None:
        super().__init__()
        self._callback = callback

    def on_modified(self, event: FileSystemEvent) -> None:
        if not event.is_directory and Path(str(event.src_path)).suffix == ".md":
            logger.info("Workspace file changed: %s", event.src_path)
            self._callback()

    def on_created(self, event: FileSystemEvent) -> None:
        if not event.is_directory and Path(str(event.src_path)).suffix == ".md":
            logger.info("Workspace file created: %s", event.src_path)
            self._callback()


class WorkspaceManager:
    """Manages the workspace memory file system.

    Provides:
    - Auto-creation of workspace directory and default files on first run
    - Loading all workspace files into memory
    - CRUD for individual files
    - Hot-reload via watchdog filesystem observer
    - Context dict for SystemPromptBuilder injection
    - SOUL.md migration from old location with backward-compatible symlink
    """

    def __init__(self, workspace_dir: str | Path | None = None) -> None:
        self._workspace_dir = Path(workspace_dir) if workspace_dir else DEFAULT_WORKSPACE_DIR
        self._files: dict[str, str] = {}
        self._observer: Observer | None = None  # pyright: ignore[reportInvalidTypeForm]
        self._lock = threading.Lock()

    @property
    def workspace_dir(self) -> Path:
        return self._workspace_dir

    @property
    def files(self) -> dict[str, str]:
        """Return a copy of all loaded workspace files."""
        with self._lock:
            return dict(self._files)

    def initialize(self) -> list[str]:
        """Create workspace directory and default files. Returns list of created filenames."""
        self._workspace_dir.mkdir(parents=True, exist_ok=True)
        created: list[str] = []
        for filename, default_content in DEFAULT_FILE_CONTENTS.items():
            path = self._workspace_dir / filename
            if not path.exists():
                path.write_text(default_content, encoding="utf-8")
                created.append(filename)
        return created

    def migrate_soul_md(self, old_soul_dir: str | Path | None = None) -> bool:
        """Move SOUL.md from old location into workspace, leaving a symlink.

        Returns True if migration was performed.
        """
        old_dir = Path(old_soul_dir) if old_soul_dir else Path.home() / ".orbiter-claw"
        old_soul = old_dir / "SOUL.md"
        new_soul = self._workspace_dir / "SOUL.md"

        # Only migrate if old file exists, is a regular file (not symlink),
        # and new location doesn't already have a file
        if not old_soul.is_file() or old_soul.is_symlink():
            return False
        if new_soul.exists():
            return False

        self._workspace_dir.mkdir(parents=True, exist_ok=True)

        # Copy content to new location
        content = old_soul.read_text(encoding="utf-8")
        new_soul.write_text(content, encoding="utf-8")

        # Replace old file with symlink
        old_soul.unlink()
        old_soul.symlink_to(new_soul)

        logger.info("Migrated SOUL.md from %s to %s (symlink left)", old_dir, self._workspace_dir)
        return True

    def load_all(self) -> dict[str, str]:
        """Load all workspace .md files from disk. Returns filename -> content dict."""
        with self._lock:
            self._files = {}
            if self._workspace_dir.is_dir():
                for filename in WORKSPACE_FILE_NAMES:
                    path = self._workspace_dir / filename
                    if path.is_file():
                        try:
                            self._files[filename] = path.read_text(encoding="utf-8")
                        except OSError:
                            logger.warning("Failed to read workspace file: %s", path)
                # Also load any extra .md files not in the standard list
                for path in sorted(self._workspace_dir.glob("*.md")):
                    if path.name not in self._files:
                        try:
                            self._files[path.name] = path.read_text(encoding="utf-8")
                        except OSError:
                            logger.warning("Failed to read workspace file: %s", path)
            return dict(self._files)

    def read_file(self, filename: str) -> str | None:
        """Read a single workspace file from disk. Returns content or None if not found."""
        path = self._workspace_dir / filename
        if not path.is_file():
            return None
        # Ensure we don't allow path traversal
        try:
            path.resolve().relative_to(self._workspace_dir.resolve())
        except ValueError:
            return None
        try:
            content = path.read_text(encoding="utf-8")
            with self._lock:
                self._files[filename] = content
            return content
        except OSError:
            return None

    def write_file(self, filename: str, content: str) -> bool:
        """Write content to a workspace file. Returns True on success."""
        # Ensure we don't allow path traversal
        if "/" in filename or "\\" in filename or ".." in filename:
            return False
        self._workspace_dir.mkdir(parents=True, exist_ok=True)
        path = self._workspace_dir / filename
        try:
            path.write_text(content, encoding="utf-8")
            with self._lock:
                self._files[filename] = content
            return True
        except OSError:
            logger.warning("Failed to write workspace file: %s", path)
            return False

    def list_files(self) -> list[dict[str, object]]:
        """List all .md files in workspace with metadata."""
        if not self._workspace_dir.is_dir():
            return []
        result: list[dict[str, object]] = []
        for path in sorted(self._workspace_dir.glob("*.md")):
            try:
                stat = path.stat()
                result.append(
                    {
                        "filename": path.name,
                        "size": stat.st_size,
                        "modified": stat.st_mtime,
                    }
                )
            except OSError:
                continue
        return result

    def context_for_prompt(self) -> dict[str, str]:
        """Return workspace files as a context_files dict for SystemPromptBuilder.

        Only includes non-empty files.
        """
        with self._lock:
            return {name: content for name, content in self._files.items() if content.strip()}

    def start_watching(self) -> None:
        """Start filesystem watcher for workspace file changes."""
        if self._observer is not None:
            return

        self._workspace_dir.mkdir(parents=True, exist_ok=True)

        def _reload() -> None:
            self.load_all()

        handler = _WorkspaceFileHandler(_reload)
        observer = Observer()  # pyright: ignore[reportMissingTypeStubs]
        observer.schedule(handler, str(self._workspace_dir), recursive=False)  # pyright: ignore[reportUnknownMemberType]
        observer.daemon = True  # pyright: ignore[reportUnknownMemberType]
        observer.start()  # pyright: ignore[reportUnknownMemberType]
        self._observer = observer
        logger.info("Started watching workspace at %s", self._workspace_dir)

    def stop_watching(self) -> None:
        """Stop filesystem watcher."""
        observer = self._observer
        if observer is not None:
            observer.stop()  # pyright: ignore[reportUnknownMemberType]
            observer.join(timeout=5)  # pyright: ignore[reportUnknownMemberType]
            self._observer = None
            logger.info("Stopped watching workspace")


# Module-level singleton
_workspace: WorkspaceManager | None = None


def get_workspace() -> WorkspaceManager:
    """Get the module-level WorkspaceManager singleton.

    Creates and initializes one if it doesn't exist.
    """
    global _workspace  # noqa: PLW0603
    if _workspace is None:
        _workspace = WorkspaceManager()
        _workspace.initialize()
        _workspace.migrate_soul_md()
        _workspace.load_all()
    return _workspace


def init_workspace(manager: WorkspaceManager) -> None:
    """Set the module-level WorkspaceManager singleton."""
    global _workspace  # noqa: PLW0603
    _workspace = manager
