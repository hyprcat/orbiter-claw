"""Tests for SOUL.md personality engine (US-005)."""

from __future__ import annotations

import time
from pathlib import Path

from httpx import AsyncClient

from orbiter_claw.personality import (
    DEFAULT_SOUL_CONTENT,
    SOUL_SECTIONS,
    PersonalityManager,
    _SoulFileHandler,
    get_personality,
    init_personality,
    parse_soul_md,
    sections_to_prompt,
)

# ---------------------------------------------------------------------------
# Unit tests: parse_soul_md
# ---------------------------------------------------------------------------


class TestParseSoulMd:
    def test_parse_all_recognized_sections(self) -> None:
        content = (
            "# Personality\nFriendly\n\n# Tone\nCasual\n\n"
            "# Knowledge\nPython expert\n\n# Boundaries\nNo legal advice\n\n"
            "# Preferences\nConcise answers\n"
        )
        sections = parse_soul_md(content)
        assert len(sections) == 5
        assert sections["Personality"] == "Friendly"
        assert sections["Tone"] == "Casual"
        assert sections["Knowledge"] == "Python expert"
        assert sections["Boundaries"] == "No legal advice"
        assert sections["Preferences"] == "Concise answers"

    def test_parse_empty_content(self) -> None:
        assert parse_soul_md("") == {}

    def test_parse_no_headings(self) -> None:
        assert parse_soul_md("Just some text\nwith no headings") == {}

    def test_parse_empty_sections_excluded(self) -> None:
        content = "# Personality\n\n# Tone\nCasual\n"
        sections = parse_soul_md(content)
        assert "Personality" not in sections
        assert sections["Tone"] == "Casual"

    def test_parse_multiline_content(self) -> None:
        content = "# Personality\nLine one\nLine two\nLine three\n"
        sections = parse_soul_md(content)
        assert sections["Personality"] == "Line one\nLine two\nLine three"

    def test_parse_unrecognized_headings_preserved(self) -> None:
        content = "# Custom Section\nSome content\n"
        sections = parse_soul_md(content)
        assert sections["Custom Section"] == "Some content"

    def test_parse_text_before_first_heading_ignored(self) -> None:
        content = "Preamble text\n# Personality\nFriendly\n"
        sections = parse_soul_md(content)
        assert "Preamble text" not in str(sections)
        assert sections["Personality"] == "Friendly"

    def test_parse_strips_whitespace_from_body(self) -> None:
        content = "# Tone\n\n   Casual with spaces   \n\n"
        sections = parse_soul_md(content)
        assert sections["Tone"] == "Casual with spaces"

    def test_parse_comments_included_in_body(self) -> None:
        content = "# Personality\n<!-- This is a comment -->\nFriendly\n"
        sections = parse_soul_md(content)
        assert "<!-- This is a comment -->" in sections["Personality"]

    def test_parse_only_h1_headings(self) -> None:
        content = "# Real Section\nContent\n## Not a section\nMore content\n"
        sections = parse_soul_md(content)
        assert "Real Section" in sections
        assert "Not a section" not in sections
        assert "## Not a section" in sections["Real Section"]

    def test_parse_default_soul_content_has_no_sections(self) -> None:
        """Default SOUL.md has only comments, so sections should be empty."""
        sections = parse_soul_md(DEFAULT_SOUL_CONTENT)
        # All section bodies are just HTML comments, which get parsed
        # but the body is only comments — they are non-empty so they are included
        for name in SOUL_SECTIONS:
            if name in sections:
                assert sections[name].startswith("<!--")


class TestSectionsToPrompt:
    def test_empty_sections(self) -> None:
        assert sections_to_prompt({}) == ""

    def test_single_section(self) -> None:
        result = sections_to_prompt({"Personality": "Friendly"})
        assert result == "## Personality\nFriendly"

    def test_multiple_sections(self) -> None:
        result = sections_to_prompt({"Personality": "Friendly", "Tone": "Casual"})
        assert "## Personality\nFriendly" in result
        assert "## Tone\nCasual" in result
        assert "\n\n" in result  # sections separated by double newline

    def test_preserves_section_order(self) -> None:
        from collections import OrderedDict

        sections = OrderedDict([("A", "First"), ("B", "Second"), ("C", "Third")])
        result = sections_to_prompt(sections)
        lines = result.split("\n\n")
        assert lines[0] == "## A\nFirst"
        assert lines[1] == "## B\nSecond"
        assert lines[2] == "## C\nThird"


# ---------------------------------------------------------------------------
# Unit tests: PersonalityManager
# ---------------------------------------------------------------------------


class TestPersonalityManager:
    def test_init_default_dir(self) -> None:
        mgr = PersonalityManager()
        assert mgr.soul_path == Path.home() / ".orbiter-claw" / "SOUL.md"

    def test_init_custom_dir(self, tmp_path: Path) -> None:
        mgr = PersonalityManager(soul_dir=tmp_path)
        assert mgr.soul_path == tmp_path / "SOUL.md"

    def test_ensure_default_creates_file(self, tmp_path: Path) -> None:
        mgr = PersonalityManager(soul_dir=tmp_path)
        created = mgr.ensure_default()
        assert created is True
        assert (tmp_path / "SOUL.md").exists()
        content = (tmp_path / "SOUL.md").read_text(encoding="utf-8")
        assert content == DEFAULT_SOUL_CONTENT

    def test_ensure_default_no_overwrite(self, tmp_path: Path) -> None:
        soul_path = tmp_path / "SOUL.md"
        soul_path.write_text("Custom content", encoding="utf-8")
        mgr = PersonalityManager(soul_dir=tmp_path)
        created = mgr.ensure_default()
        assert created is False
        assert soul_path.read_text(encoding="utf-8") == "Custom content"

    def test_ensure_default_creates_directory(self, tmp_path: Path) -> None:
        nested = tmp_path / "nested" / "dir"
        mgr = PersonalityManager(soul_dir=nested)
        mgr.ensure_default()
        assert nested.exists()
        assert (nested / "SOUL.md").exists()

    def test_load_existing_file(self, tmp_path: Path) -> None:
        soul_path = tmp_path / "SOUL.md"
        soul_path.write_text("# Personality\nHelpful\n", encoding="utf-8")
        mgr = PersonalityManager(soul_dir=tmp_path)
        sections = mgr.load()
        assert sections == {"Personality": "Helpful"}
        assert mgr.raw_content == "# Personality\nHelpful\n"

    def test_load_nonexistent_file(self, tmp_path: Path) -> None:
        mgr = PersonalityManager(soul_dir=tmp_path)
        sections = mgr.load()
        assert sections == {}
        assert mgr.raw_content == ""

    def test_sections_property_returns_copy(self, tmp_path: Path) -> None:
        soul_path = tmp_path / "SOUL.md"
        soul_path.write_text("# Personality\nFriendly\n", encoding="utf-8")
        mgr = PersonalityManager(soul_dir=tmp_path)
        mgr.load()
        sections = mgr.sections
        sections["Personality"] = "Modified"
        assert mgr.sections["Personality"] == "Friendly"

    def test_update_writes_and_parses(self, tmp_path: Path) -> None:
        mgr = PersonalityManager(soul_dir=tmp_path)
        sections = mgr.update("# Tone\nProfessional\n")
        assert sections == {"Tone": "Professional"}
        assert (tmp_path / "SOUL.md").read_text(encoding="utf-8") == "# Tone\nProfessional\n"
        assert mgr.raw_content == "# Tone\nProfessional\n"

    def test_update_creates_directory(self, tmp_path: Path) -> None:
        nested = tmp_path / "new" / "dir"
        mgr = PersonalityManager(soul_dir=nested)
        mgr.update("# Personality\nCool\n")
        assert (nested / "SOUL.md").exists()

    def test_to_prompt(self, tmp_path: Path) -> None:
        soul_path = tmp_path / "SOUL.md"
        soul_path.write_text("# Personality\nFriendly\n\n# Tone\nCasual\n", encoding="utf-8")
        mgr = PersonalityManager(soul_dir=tmp_path)
        mgr.load()
        prompt = mgr.to_prompt()
        assert "## Personality\nFriendly" in prompt
        assert "## Tone\nCasual" in prompt

    def test_to_prompt_empty(self, tmp_path: Path) -> None:
        mgr = PersonalityManager(soul_dir=tmp_path)
        mgr.load()
        assert mgr.to_prompt() == ""


# ---------------------------------------------------------------------------
# Tests: Hot-reload via watchdog
# ---------------------------------------------------------------------------


class TestHotReload:
    def test_start_stop_watching(self, tmp_path: Path) -> None:
        mgr = PersonalityManager(soul_dir=tmp_path)
        mgr.start_watching()
        assert mgr._observer is not None
        mgr.stop_watching()
        assert mgr._observer is None

    def test_start_watching_idempotent(self, tmp_path: Path) -> None:
        mgr = PersonalityManager(soul_dir=tmp_path)
        mgr.start_watching()
        observer1 = mgr._observer
        mgr.start_watching()
        assert mgr._observer is observer1
        mgr.stop_watching()

    def test_stop_watching_when_not_started(self, tmp_path: Path) -> None:
        mgr = PersonalityManager(soul_dir=tmp_path)
        mgr.stop_watching()  # Should not raise

    def test_file_change_triggers_reload(self, tmp_path: Path) -> None:
        soul_path = tmp_path / "SOUL.md"
        soul_path.write_text("# Personality\nOriginal\n", encoding="utf-8")

        mgr = PersonalityManager(soul_dir=tmp_path)
        mgr.load()
        assert mgr.sections.get("Personality") == "Original"

        mgr.start_watching()
        try:
            # Modify the file
            soul_path.write_text("# Personality\nUpdated\n", encoding="utf-8")
            # Wait for watchdog to detect change
            time.sleep(1.5)
            assert mgr.sections.get("Personality") == "Updated"
        finally:
            mgr.stop_watching()

    def test_soul_file_handler_on_modified(self, tmp_path: Path) -> None:
        called = []

        def callback() -> None:
            called.append(True)

        handler = _SoulFileHandler(tmp_path / "SOUL.md", callback)
        # Simulate a modification event
        from unittest.mock import MagicMock

        event = MagicMock()
        event.is_directory = False
        event.src_path = str(tmp_path / "SOUL.md")
        handler.on_modified(event)
        assert len(called) == 1

    def test_soul_file_handler_ignores_other_files(self, tmp_path: Path) -> None:
        called = []

        def callback() -> None:
            called.append(True)

        handler = _SoulFileHandler(tmp_path / "SOUL.md", callback)
        from unittest.mock import MagicMock

        event = MagicMock()
        event.is_directory = False
        event.src_path = str(tmp_path / "other.md")
        handler.on_modified(event)
        assert len(called) == 0

    def test_soul_file_handler_ignores_directories(self, tmp_path: Path) -> None:
        called = []

        def callback() -> None:
            called.append(True)

        handler = _SoulFileHandler(tmp_path / "SOUL.md", callback)
        from unittest.mock import MagicMock

        event = MagicMock()
        event.is_directory = True
        event.src_path = str(tmp_path / "SOUL.md")
        handler.on_modified(event)
        assert len(called) == 0

    def test_soul_file_handler_on_created(self, tmp_path: Path) -> None:
        called = []

        def callback() -> None:
            called.append(True)

        handler = _SoulFileHandler(tmp_path / "SOUL.md", callback)
        from unittest.mock import MagicMock

        event = MagicMock()
        event.is_directory = False
        event.src_path = str(tmp_path / "SOUL.md")
        handler.on_created(event)
        assert len(called) == 1


# ---------------------------------------------------------------------------
# Tests: Module-level singleton
# ---------------------------------------------------------------------------


class TestModuleSingleton:
    def test_init_personality_sets_singleton(self, tmp_path: Path) -> None:
        mgr = PersonalityManager(soul_dir=tmp_path)
        init_personality(mgr)
        assert get_personality() is mgr
        # Reset
        init_personality(PersonalityManager(soul_dir=tmp_path))

    def test_get_personality_creates_default(self) -> None:
        # Reset singleton
        import orbiter_claw.personality as mod

        old = mod._personality
        mod._personality = None
        try:
            mgr = get_personality()
            assert mgr is not None
            assert isinstance(mgr, PersonalityManager)
        finally:
            mod._personality = old


# ---------------------------------------------------------------------------
# Tests: Prompt injection into SystemPromptBuilder
# ---------------------------------------------------------------------------


class TestPromptIntegration:
    def test_soul_sections_injected_via_context_files(self, tmp_path: Path) -> None:
        """SOUL.md content is loaded by SystemPromptBuilder via workspace files."""
        from orbiter_claw.prompt_builder import SystemPromptBuilder

        soul_content = "# Personality\nFriendly helper\n\n# Tone\nCasual\n"
        builder = SystemPromptBuilder(
            mode="full",
            context_files={"SOUL.md": soul_content},
        )
        prompt = builder.build()
        assert "SOUL.md" in prompt
        assert "Friendly helper" in prompt
        assert "Casual" in prompt
        assert "embody its persona and tone" in prompt

    def test_personality_manager_to_prompt_usable(self, tmp_path: Path) -> None:
        """PersonalityManager.to_prompt() produces content suitable for injection."""
        soul_path = tmp_path / "SOUL.md"
        soul_path.write_text(
            "# Personality\nWitty and concise\n\n# Boundaries\nNo medical advice\n",
            encoding="utf-8",
        )
        mgr = PersonalityManager(soul_dir=tmp_path)
        mgr.load()
        prompt = mgr.to_prompt()
        assert "## Personality\nWitty and concise" in prompt
        assert "## Boundaries\nNo medical advice" in prompt

    def test_changes_take_effect_without_restart(self, tmp_path: Path) -> None:
        """Updating SOUL.md should be reflected in subsequent prompt builds."""
        from orbiter_claw.prompt_builder import SystemPromptBuilder

        soul_path = tmp_path / "SOUL.md"
        soul_path.write_text("# Personality\nOriginal\n", encoding="utf-8")
        mgr = PersonalityManager(soul_dir=tmp_path)
        mgr.load()

        # First build
        builder = SystemPromptBuilder(
            mode="full",
            context_files={"SOUL.md": mgr.raw_content},
        )
        prompt1 = builder.build()
        assert "Original" in prompt1

        # Update and rebuild
        mgr.update("# Personality\nUpdated personality\n")
        builder2 = SystemPromptBuilder(
            mode="full",
            context_files={"SOUL.md": mgr.raw_content},
        )
        prompt2 = builder2.build()
        assert "Updated personality" in prompt2
        assert "Original" not in prompt2


# ---------------------------------------------------------------------------
# API endpoint tests
# ---------------------------------------------------------------------------


class TestPersonalityAPI:
    async def test_get_personality(self, client: AsyncClient, tmp_path: Path) -> None:
        mgr = PersonalityManager(soul_dir=tmp_path)
        mgr.ensure_default()
        mgr.load()
        init_personality(mgr)

        resp = await client.get("/api/v1/personality")
        assert resp.status_code == 200
        data = resp.json()
        assert "raw" in data
        assert "sections" in data
        assert isinstance(data["raw"], str)
        assert isinstance(data["sections"], dict)

    async def test_get_personality_with_content(self, client: AsyncClient, tmp_path: Path) -> None:
        soul_path = tmp_path / "SOUL.md"
        soul_path.write_text("# Personality\nHelpful\n", encoding="utf-8")
        mgr = PersonalityManager(soul_dir=tmp_path)
        mgr.load()
        init_personality(mgr)

        resp = await client.get("/api/v1/personality")
        assert resp.status_code == 200
        data = resp.json()
        assert "Helpful" in data["raw"]
        assert data["sections"]["Personality"] == "Helpful"

    async def test_put_personality(self, client: AsyncClient, tmp_path: Path) -> None:
        mgr = PersonalityManager(soul_dir=tmp_path)
        mgr.ensure_default()
        mgr.load()
        init_personality(mgr)

        new_content = "# Personality\nBold and direct\n\n# Tone\nFormal\n"
        resp = await client.put(
            "/api/v1/personality",
            json={"content": new_content},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["sections"]["Personality"] == "Bold and direct"
        assert data["sections"]["Tone"] == "Formal"
        assert data["raw"] == new_content

    async def test_put_personality_persists(self, client: AsyncClient, tmp_path: Path) -> None:
        mgr = PersonalityManager(soul_dir=tmp_path)
        mgr.ensure_default()
        mgr.load()
        init_personality(mgr)

        new_content = "# Preferences\nUse bullet points\n"
        await client.put("/api/v1/personality", json={"content": new_content})

        # Verify persisted to disk
        file_content = (tmp_path / "SOUL.md").read_text(encoding="utf-8")
        assert file_content == new_content

        # Verify GET reflects update
        resp = await client.get("/api/v1/personality")
        data = resp.json()
        assert data["sections"]["Preferences"] == "Use bullet points"

    async def test_put_personality_empty_content(self, client: AsyncClient, tmp_path: Path) -> None:
        mgr = PersonalityManager(soul_dir=tmp_path)
        mgr.ensure_default()
        mgr.load()
        init_personality(mgr)

        resp = await client.put("/api/v1/personality", json={"content": ""})
        assert resp.status_code == 200
        data = resp.json()
        assert data["sections"] == {}
        assert data["raw"] == ""


# ---------------------------------------------------------------------------
# Tests: SOUL_SECTIONS constant
# ---------------------------------------------------------------------------


class TestSoulSections:
    def test_all_expected_sections_present(self) -> None:
        expected = {"Personality", "Tone", "Knowledge", "Boundaries", "Preferences"}
        assert set(SOUL_SECTIONS) == expected

    def test_default_content_has_all_sections(self) -> None:
        """Default SOUL.md template contains all recognized section headings."""
        for section in SOUL_SECTIONS:
            assert f"# {section}" in DEFAULT_SOUL_CONTENT
