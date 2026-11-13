"""SOUL.md personality engine.

Parses the SOUL.md file into structured sections and provides hot-reload
on file change via watchdog filesystem observer. Sections are injected into
the agent's system prompt through the SystemPromptBuilder.
"""

from __future__ import annotations

import logging
import re
import threading
from pathlib import Path
from typing import TYPE_CHECKING

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer  # pyright: ignore[reportMissingTypeStubs]

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)

DEFAULT_SOUL_DIR = Path.home() / ".orbiter-claw"

# Recognized SOUL.md section headings
SOUL_SECTIONS = ("Personality", "Tone", "Knowledge", "Boundaries", "Preferences")

DEFAULT_SOUL_CONTENT = """\
# Personality
<!-- Define how your assistant should behave. Example: -->
<!-- Friendly and casual, uses humor occasionally -->

# Tone
<!-- Set the communication style. Example: -->
<!-- Conversational but precise, avoids jargon unless asked -->

# Knowledge
<!-- Specify areas of expertise or context. Example: -->
<!-- Expert in Python, familiar with web development -->

# Boundaries
<!-- Define what the assistant should avoid. Example: -->
<!-- Never give medical or legal advice, always disclaim uncertainty -->

# Preferences
<!-- Personal preferences for responses. Example: -->
<!-- Prefer concise answers, use bullet points for lists -->
"""


def parse_soul_md(content: str) -> dict[str, str]:
    """Parse SOUL.md content into a dict of section_name -> section_content.

    Recognized sections: Personality, Tone, Knowledge, Boundaries, Preferences.
    Unrecognized headings are preserved under their heading name.
    Lines before the first heading are ignored.
    """
    sections: dict[str, str] = {}
    current_section: str | None = None
    current_lines: list[str] = []

    for line in content.splitlines():
        heading_match = re.match(r"^#\s+(.+)$", line)
        if heading_match:
            # Save previous section
            if current_section is not None:
                body = "\n".join(current_lines).strip()
                if body:
                    sections[current_section] = body
            current_section = heading_match.group(1).strip()
            current_lines = []
        elif current_section is not None:
            current_lines.append(line)

    # Save last section
    if current_section is not None:
        body = "\n".join(current_lines).strip()
        if body:
            sections[current_section] = body

    return sections


def sections_to_prompt(sections: dict[str, str]) -> str:
    """Convert parsed sections into a prompt string for system prompt injection."""
    if not sections:
        return ""
    parts: list[str] = []
    for name, content in sections.items():
        parts.append(f"## {name}\n{content}")
    return "\n\n".join(parts)


class _SoulFileHandler(FileSystemEventHandler):
    """Watches for SOUL.md changes and triggers a callback."""

    def __init__(self, soul_path: Path, callback: Callable[[], None]) -> None:
        super().__init__()
        self._soul_path = soul_path
        self._callback = callback

    def on_modified(self, event: FileSystemEvent) -> None:
        if not event.is_directory and Path(str(event.src_path)).name == "SOUL.md":
            logger.info("SOUL.md changed, reloading personality")
            self._callback()

    def on_created(self, event: FileSystemEvent) -> None:
        if not event.is_directory and Path(str(event.src_path)).name == "SOUL.md":
            logger.info("SOUL.md created, loading personality")
            self._callback()


class PersonalityManager:
    """Manages the SOUL.md personality file.

    Provides:
    - Loading and parsing SOUL.md into sections
    - Hot-reload via watchdog filesystem observer
    - Default file generation on first run
    - Read/update API for the file content
    """

    def __init__(self, soul_dir: str | Path | None = None) -> None:
        self._soul_dir = Path(soul_dir) if soul_dir else DEFAULT_SOUL_DIR
        self._soul_path = self._soul_dir / "SOUL.md"
        self._sections: dict[str, str] = {}
        self._raw_content: str = ""
        self._observer: Observer | None = None  # pyright: ignore[reportInvalidTypeForm]
        self._lock = threading.Lock()

    @property
    def soul_path(self) -> Path:
        return self._soul_path

    @property
    def sections(self) -> dict[str, str]:
        with self._lock:
            return dict(self._sections)

    @property
    def raw_content(self) -> str:
        with self._lock:
            return self._raw_content

    def ensure_default(self) -> bool:
        """Create default SOUL.md if it doesn't exist. Returns True if created."""
        self._soul_dir.mkdir(parents=True, exist_ok=True)
        if not self._soul_path.exists():
            self._soul_path.write_text(DEFAULT_SOUL_CONTENT, encoding="utf-8")
            return True
        return False

    def load(self) -> dict[str, str]:
        """Load and parse SOUL.md from disk. Returns parsed sections."""
        with self._lock:
            if self._soul_path.is_file():
                self._raw_content = self._soul_path.read_text(encoding="utf-8")
            else:
                self._raw_content = ""
            self._sections = parse_soul_md(self._raw_content)
            return dict(self._sections)

    def update(self, content: str) -> dict[str, str]:
        """Write new content to SOUL.md and return parsed sections."""
        self._soul_dir.mkdir(parents=True, exist_ok=True)
        self._soul_path.write_text(content, encoding="utf-8")
        return self.load()

    def to_prompt(self) -> str:
        """Get the current personality as a prompt string."""
        return sections_to_prompt(self.sections)

    def start_watching(self) -> None:
        """Start filesystem watcher for SOUL.md changes."""
        if self._observer is not None:
            return

        self._soul_dir.mkdir(parents=True, exist_ok=True)

        def _reload() -> None:
            self.load()

        handler = _SoulFileHandler(self._soul_path, _reload)
        observer = Observer()  # pyright: ignore[reportMissingTypeStubs]
        observer.schedule(handler, str(self._soul_dir), recursive=False)  # pyright: ignore[reportUnknownMemberType]
        observer.daemon = True  # pyright: ignore[reportUnknownMemberType]
        observer.start()  # pyright: ignore[reportUnknownMemberType]
        self._observer = observer
        logger.info("Started watching %s for SOUL.md changes", self._soul_dir)

    def stop_watching(self) -> None:
        """Stop filesystem watcher."""
        observer = self._observer
        if observer is not None:
            observer.stop()  # pyright: ignore[reportUnknownMemberType]
            observer.join(timeout=5)  # pyright: ignore[reportUnknownMemberType]
            self._observer = None
            logger.info("Stopped watching for SOUL.md changes")


# Module-level singleton
_personality: PersonalityManager | None = None


def get_personality() -> PersonalityManager:
    """Get the module-level PersonalityManager singleton.

    Creates and initializes one if it doesn't exist. Call init_personality()
    to set a custom instance.
    """
    global _personality  # noqa: PLW0603
    if _personality is None:
        _personality = PersonalityManager()
        _personality.ensure_default()
        _personality.load()
    return _personality


def init_personality(manager: PersonalityManager) -> None:
    """Set the module-level PersonalityManager singleton."""
    global _personality  # noqa: PLW0603
    _personality = manager
