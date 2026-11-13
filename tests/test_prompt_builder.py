"""Tests for the SystemPromptBuilder."""

from __future__ import annotations

from pathlib import Path

from orbiter_claw.prompt_builder import (
    HEARTBEAT_OK_TOKEN,
    SILENT_REPLY_TOKEN,
    SystemPromptBuilder,
    sanitize_prompt_content,
)

# --- sanitize_prompt_content ---


class TestSanitizePromptContent:
    def test_strips_null_bytes(self) -> None:
        assert sanitize_prompt_content("hello\x00world") == "helloworld"

    def test_strips_c0_controls(self) -> None:
        # \x01-\x08, \x0e-\x1f are C0 controls (excluding \t, \n, \r)
        result = sanitize_prompt_content("a\x01b\x02c\x07d\x1fe")
        assert result == "abcde"

    def test_strips_c1_controls(self) -> None:
        result = sanitize_prompt_content("hello\x80\x8f\x9fworld")
        assert result == "helloworld"

    def test_preserves_normal_whitespace(self) -> None:
        text = "line1\nline2\tindented\rcarriage"
        assert sanitize_prompt_content(text) == text

    def test_preserves_unicode_text(self) -> None:
        text = "Hello 世界 🌍 café"
        assert sanitize_prompt_content(text) == text

    def test_empty_string(self) -> None:
        assert sanitize_prompt_content("") == ""

    def test_strips_vertical_tab_and_form_feed(self) -> None:
        # \x0b (VT) and \x0c (FF) are stripped
        result = sanitize_prompt_content("a\x0bb\x0cc")
        assert result == "abc"


# --- PromptMode "none" ---


class TestPromptModeNone:
    def test_returns_identity_line_only(self) -> None:
        builder = SystemPromptBuilder(mode="none")
        result = builder.build()
        assert result == "You are a personal AI assistant running inside orbiter-claw."

    def test_ignores_all_other_params(self) -> None:
        builder = SystemPromptBuilder(
            mode="none",
            tool_names=["read", "write"],
            skills_prompt="some skills",
            heartbeat_prompt="check things",
            model="openai:gpt-4o",
            channel="telegram",
        )
        result = builder.build()
        assert result == "You are a personal AI assistant running inside orbiter-claw."


# --- PromptMode "full" ---


class TestPromptModeFull:
    def test_contains_identity(self) -> None:
        builder = SystemPromptBuilder(mode="full")
        result = builder.build()
        assert "You are a personal AI assistant running inside orbiter-claw." in result

    def test_contains_tooling_section(self) -> None:
        builder = SystemPromptBuilder(mode="full", tool_names=["read", "exec"])
        result = builder.build()
        assert "## Tooling" in result
        assert "- read" in result
        assert "- exec" in result

    def test_tool_summaries_included(self) -> None:
        builder = SystemPromptBuilder(
            mode="full",
            tool_names=["read"],
            tool_summaries={"read": "Read file contents"},
        )
        result = builder.build()
        assert "- read: Read file contents" in result

    def test_no_tools_shows_placeholder(self) -> None:
        builder = SystemPromptBuilder(mode="full", tool_names=[])
        result = builder.build()
        assert "(No tools currently registered)" in result

    def test_contains_tool_call_style(self) -> None:
        builder = SystemPromptBuilder(mode="full")
        result = builder.build()
        assert "## Tool Call Style" in result
        assert "do not narrate routine" in result

    def test_contains_safety_section(self) -> None:
        builder = SystemPromptBuilder(mode="full")
        result = builder.build()
        assert "## Safety" in result
        assert "self-preservation" in result
        assert "replication" in result
        assert "data" in result
        assert "trash" in result

    def test_contains_memory_recall(self) -> None:
        builder = SystemPromptBuilder(mode="full")
        result = builder.build()
        assert "## Memory Recall" in result
        assert "MEMORY.md" in result

    def test_contains_workspace_info(self) -> None:
        builder = SystemPromptBuilder(mode="full")
        result = builder.build()
        assert "## Workspace" in result
        assert "~/.orbiter-claw/workspace" in result

    def test_custom_workspace_dir(self) -> None:
        builder = SystemPromptBuilder(mode="full", workspace_dir="/tmp/my-workspace")
        result = builder.build()
        assert "/tmp/my-workspace" in result

    def test_contains_datetime_section(self) -> None:
        builder = SystemPromptBuilder(mode="full")
        result = builder.build()
        assert "## Current Date & Time" in result
        assert "UTC:" in result

    def test_user_timezone_in_datetime(self) -> None:
        builder = SystemPromptBuilder(mode="full", user_timezone="America/New_York")
        result = builder.build()
        assert "America/New_York" in result

    def test_contains_messaging(self) -> None:
        builder = SystemPromptBuilder(mode="full", channel="telegram")
        result = builder.build()
        assert "## Messaging" in result
        assert "telegram" in result

    def test_contains_silent_replies(self) -> None:
        builder = SystemPromptBuilder(mode="full")
        result = builder.build()
        assert "## Silent Replies" in result
        assert SILENT_REPLY_TOKEN in result

    def test_contains_heartbeat(self) -> None:
        builder = SystemPromptBuilder(mode="full")
        result = builder.build()
        assert "## Heartbeats" in result
        assert HEARTBEAT_OK_TOKEN in result

    def test_custom_heartbeat_prompt(self) -> None:
        builder = SystemPromptBuilder(mode="full", heartbeat_prompt="Check emails and calendar")
        result = builder.build()
        assert "Check emails and calendar" in result

    def test_contains_runtime(self) -> None:
        builder = SystemPromptBuilder(mode="full", model="openai:gpt-4o")
        result = builder.build()
        assert "## Runtime" in result
        assert "model=openai:gpt-4o" in result

    def test_runtime_with_channel_and_capabilities(self) -> None:
        builder = SystemPromptBuilder(
            mode="full",
            channel="telegram",
            capabilities=["inlineButtons", "reactions"],
        )
        result = builder.build()
        assert "channel=telegram" in result
        assert "capabilities=inlineButtons,reactions" in result

    def test_runtime_no_capabilities_shows_none(self) -> None:
        builder = SystemPromptBuilder(mode="full", channel="telegram")
        result = builder.build()
        assert "capabilities=none" in result

    def test_skills_prompt_included(self) -> None:
        skills = "<available_skills>\nweather: Check weather\n</available_skills>"
        builder = SystemPromptBuilder(mode="full", skills_prompt=skills)
        result = builder.build()
        assert "## Skills" in result
        assert "weather: Check weather" in result

    def test_empty_skills_prompt_excluded(self) -> None:
        builder = SystemPromptBuilder(mode="full", skills_prompt="  ")
        result = builder.build()
        assert "## Skills" not in result

    def test_extra_system_prompt_full_mode(self) -> None:
        builder = SystemPromptBuilder(mode="full", extra_system_prompt="You are in a group chat.")
        result = builder.build()
        assert "## Additional Context" in result
        assert "You are in a group chat." in result

    def test_section_ordering(self) -> None:
        """Verify sections appear in the correct order."""
        builder = SystemPromptBuilder(
            mode="full",
            tool_names=["read"],
            skills_prompt="some skills",
            model="openai:gpt-4o",
        )
        result = builder.build()
        sections = [
            "## Tooling",
            "## Tool Call Style",
            "## Safety",
            "## Skills",
            "## Memory Recall",
            "## Workspace",
            "## Current Date & Time",
            "## Messaging",
            "## Silent Replies",
            "## Heartbeats",
            "## Runtime",
        ]
        positions = []
        for section in sections:
            pos = result.find(section)
            if pos >= 0:
                positions.append(pos)
        # All found sections should be in ascending order
        assert positions == sorted(positions), (
            f"Sections out of order: {list(zip(sections, positions, strict=False))}"
        )


# --- PromptMode "minimal" ---


class TestPromptModeMinimal:
    def test_contains_identity(self) -> None:
        builder = SystemPromptBuilder(mode="minimal")
        result = builder.build()
        assert "You are a personal AI assistant running inside orbiter-claw." in result

    def test_contains_tooling(self) -> None:
        builder = SystemPromptBuilder(mode="minimal", tool_names=["read"])
        result = builder.build()
        assert "## Tooling" in result
        assert "- read" in result

    def test_contains_workspace(self) -> None:
        builder = SystemPromptBuilder(mode="minimal")
        result = builder.build()
        assert "## Workspace" in result

    def test_contains_runtime(self) -> None:
        builder = SystemPromptBuilder(mode="minimal", model="openai:gpt-4o")
        result = builder.build()
        assert "## Runtime" in result

    def test_excludes_tool_call_style(self) -> None:
        builder = SystemPromptBuilder(mode="minimal")
        result = builder.build()
        assert "## Tool Call Style" not in result

    def test_excludes_safety(self) -> None:
        builder = SystemPromptBuilder(mode="minimal")
        result = builder.build()
        assert "## Safety" not in result

    def test_excludes_skills(self) -> None:
        builder = SystemPromptBuilder(mode="minimal", skills_prompt="some skills")
        result = builder.build()
        assert "## Skills" not in result

    def test_excludes_memory_recall(self) -> None:
        builder = SystemPromptBuilder(mode="minimal")
        result = builder.build()
        assert "## Memory Recall" not in result

    def test_excludes_datetime(self) -> None:
        builder = SystemPromptBuilder(mode="minimal")
        result = builder.build()
        assert "## Current Date & Time" not in result

    def test_excludes_messaging(self) -> None:
        builder = SystemPromptBuilder(mode="minimal", channel="telegram")
        result = builder.build()
        assert "## Messaging" not in result

    def test_excludes_silent_replies(self) -> None:
        builder = SystemPromptBuilder(mode="minimal")
        result = builder.build()
        assert "## Silent Replies" not in result

    def test_excludes_heartbeat(self) -> None:
        builder = SystemPromptBuilder(mode="minimal")
        result = builder.build()
        assert "## Heartbeats" not in result

    def test_extra_prompt_uses_subagent_header(self) -> None:
        builder = SystemPromptBuilder(mode="minimal", extra_system_prompt="Research task context.")
        result = builder.build()
        assert "## Subagent Context" in result
        assert "Research task context." in result


# --- Workspace file loading ---


class TestWorkspaceFileLoading:
    def test_loads_files_from_disk(self, tmp_path: Path) -> None:
        ws = tmp_path / "workspace"
        ws.mkdir()
        (ws / "SOUL.md").write_text("# Be helpful\nBe kind.")
        (ws / "USER.md").write_text("# About User\nName: Alice")

        builder = SystemPromptBuilder(mode="full", workspace_dir=ws)
        result = builder.build()

        assert "# Project Context" in result
        assert "## SOUL.md" in result
        assert "Be kind." in result
        assert "## USER.md" in result
        assert "Name: Alice" in result

    def test_soul_md_special_handling(self, tmp_path: Path) -> None:
        ws = tmp_path / "workspace"
        ws.mkdir()
        (ws / "SOUL.md").write_text("# My Soul\nBe witty.")

        builder = SystemPromptBuilder(mode="full", workspace_dir=ws)
        result = builder.build()

        assert "embody its persona and tone" in result
        assert "Avoid stiff, generic replies" in result

    def test_no_soul_md_no_special_handling(self, tmp_path: Path) -> None:
        ws = tmp_path / "workspace"
        ws.mkdir()
        (ws / "USER.md").write_text("# User info\nName: Bob")

        builder = SystemPromptBuilder(mode="full", workspace_dir=ws)
        result = builder.build()

        assert "embody its persona" not in result

    def test_empty_files_excluded(self, tmp_path: Path) -> None:
        ws = tmp_path / "workspace"
        ws.mkdir()
        (ws / "SOUL.md").write_text("")
        (ws / "USER.md").write_text("  \n  ")
        (ws / "IDENTITY.md").write_text("# Identity\nName: Claw")

        builder = SystemPromptBuilder(mode="full", workspace_dir=ws)
        result = builder.build()

        assert "## SOUL.md" not in result
        assert "## USER.md" not in result
        assert "## IDENTITY.md" in result

    def test_missing_workspace_dir(self) -> None:
        builder = SystemPromptBuilder(mode="full", workspace_dir="/nonexistent/path")
        result = builder.build()
        assert "# Project Context" not in result

    def test_no_workspace_dir(self) -> None:
        builder = SystemPromptBuilder(mode="full")
        result = builder.build()
        # No workspace_dir → no project context
        assert "# Project Context" not in result

    def test_context_files_override_disk(self) -> None:
        """When context_files dict is passed, use it instead of disk."""
        builder = SystemPromptBuilder(
            mode="full",
            context_files={
                "SOUL.md": "# Custom Soul\nBe awesome.",
                "AGENTS.md": "# Rules\nFollow them.",
            },
        )
        result = builder.build()
        assert "## SOUL.md" in result
        assert "Be awesome." in result
        assert "## AGENTS.md" in result
        assert "Follow them." in result

    def test_context_files_empty_excluded(self) -> None:
        builder = SystemPromptBuilder(
            mode="full",
            context_files={"SOUL.md": "  ", "USER.md": "# User\nAlice"},
        )
        result = builder.build()
        assert "## SOUL.md" not in result
        assert "## USER.md" in result

    def test_file_priority_order(self, tmp_path: Path) -> None:
        """Workspace files should appear in WORKSPACE_FILES order."""
        ws = tmp_path / "workspace"
        ws.mkdir()
        (ws / "TOOLS.md").write_text("# Tools\nSome tools")
        (ws / "SOUL.md").write_text("# Soul\nBe good")
        (ws / "USER.md").write_text("# User\nBob")

        builder = SystemPromptBuilder(mode="full", workspace_dir=ws)
        result = builder.build()

        soul_pos = result.find("## SOUL.md")
        user_pos = result.find("## USER.md")
        tools_pos = result.find("## TOOLS.md")

        assert soul_pos < user_pos < tools_pos, (
            "Files should appear in WORKSPACE_FILES order: SOUL.md, USER.md, TOOLS.md"
        )

    def test_sanitizes_file_content(self, tmp_path: Path) -> None:
        ws = tmp_path / "workspace"
        ws.mkdir()
        (ws / "SOUL.md").write_text("Hello\x00World\x01!")

        builder = SystemPromptBuilder(mode="full", workspace_dir=ws)
        result = builder.build()

        assert "HelloWorld!" in result
        assert "\x00" not in result
        assert "\x01" not in result

    def test_minimal_mode_loads_workspace_files(self, tmp_path: Path) -> None:
        """Minimal mode still includes workspace files."""
        ws = tmp_path / "workspace"
        ws.mkdir()
        (ws / "SOUL.md").write_text("# Soul\nBe good")

        builder = SystemPromptBuilder(mode="minimal", workspace_dir=ws)
        result = builder.build()
        assert "## SOUL.md" in result


# --- Token constants ---


class TestTokenConstants:
    def test_silent_reply_token_value(self) -> None:
        assert SILENT_REPLY_TOKEN == "<<SILENT>>"

    def test_heartbeat_ok_token_value(self) -> None:
        assert HEARTBEAT_OK_TOKEN == "HEARTBEAT_OK"


# --- Runtime info ---


class TestRuntimeInfo:
    def test_os_info_present(self) -> None:
        builder = SystemPromptBuilder(mode="full")
        result = builder.build()
        assert "os=" in result

    def test_default_model_shown(self) -> None:
        builder = SystemPromptBuilder(mode="full", default_model="anthropic:claude-3-opus")
        result = builder.build()
        assert "default_model=anthropic:claude-3-opus" in result

    def test_runtime_sanitizes_values(self) -> None:
        builder = SystemPromptBuilder(mode="full", model="test\x00model")
        result = builder.build()
        assert "model=testmodel" in result
        assert "\x00" not in result
