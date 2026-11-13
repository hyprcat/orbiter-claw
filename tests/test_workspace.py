"""Tests for workspace memory file system (US-031)."""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import MagicMock

from httpx import AsyncClient

from orbiter_claw.workspace import (
    DEFAULT_FILE_CONTENTS,
    WORKSPACE_FILE_NAMES,
    WorkspaceManager,
    _WorkspaceFileHandler,
    get_workspace,
    init_workspace,
)

# ---------------------------------------------------------------------------
# Unit tests: WorkspaceManager initialization
# ---------------------------------------------------------------------------


class TestWorkspaceInit:
    def test_default_workspace_dir(self) -> None:
        mgr = WorkspaceManager()
        assert mgr.workspace_dir == Path.home() / ".orbiter-claw" / "workspace"

    def test_custom_workspace_dir(self, tmp_path: Path) -> None:
        mgr = WorkspaceManager(workspace_dir=tmp_path)
        assert mgr.workspace_dir == tmp_path

    def test_initialize_creates_directory(self, tmp_path: Path) -> None:
        ws_dir = tmp_path / "workspace"
        mgr = WorkspaceManager(workspace_dir=ws_dir)
        mgr.initialize()
        assert ws_dir.is_dir()

    def test_initialize_creates_default_files(self, tmp_path: Path) -> None:
        mgr = WorkspaceManager(workspace_dir=tmp_path)
        created = mgr.initialize()
        for filename in DEFAULT_FILE_CONTENTS:
            assert (tmp_path / filename).exists()
            assert filename in created

    def test_initialize_no_overwrite_existing(self, tmp_path: Path) -> None:
        (tmp_path / "USER.md").write_text("My custom user file", encoding="utf-8")
        mgr = WorkspaceManager(workspace_dir=tmp_path)
        created = mgr.initialize()
        assert "USER.md" not in created
        assert (tmp_path / "USER.md").read_text(encoding="utf-8") == "My custom user file"

    def test_initialize_returns_created_filenames(self, tmp_path: Path) -> None:
        mgr = WorkspaceManager(workspace_dir=tmp_path)
        created = mgr.initialize()
        assert isinstance(created, list)
        assert set(created) == set(DEFAULT_FILE_CONTENTS.keys())

    def test_initialize_nested_directory(self, tmp_path: Path) -> None:
        nested = tmp_path / "a" / "b" / "c"
        mgr = WorkspaceManager(workspace_dir=nested)
        mgr.initialize()
        assert nested.is_dir()

    def test_initialize_idempotent(self, tmp_path: Path) -> None:
        mgr = WorkspaceManager(workspace_dir=tmp_path)
        created1 = mgr.initialize()
        created2 = mgr.initialize()
        assert len(created1) > 0
        assert len(created2) == 0  # No files created on second run


# ---------------------------------------------------------------------------
# Unit tests: SOUL.md migration
# ---------------------------------------------------------------------------


class TestSoulMdMigration:
    def test_migrate_soul_md_moves_file(self, tmp_path: Path) -> None:
        old_dir = tmp_path / "old"
        old_dir.mkdir()
        old_soul = old_dir / "SOUL.md"
        old_soul.write_text("# Personality\nFriendly\n", encoding="utf-8")

        ws_dir = tmp_path / "workspace"
        mgr = WorkspaceManager(workspace_dir=ws_dir)
        result = mgr.migrate_soul_md(old_soul_dir=old_dir)

        assert result is True
        assert (ws_dir / "SOUL.md").exists()
        assert (ws_dir / "SOUL.md").read_text(encoding="utf-8") == "# Personality\nFriendly\n"
        assert old_soul.is_symlink()
        assert old_soul.resolve() == (ws_dir / "SOUL.md").resolve()

    def test_migrate_soul_md_no_old_file(self, tmp_path: Path) -> None:
        old_dir = tmp_path / "old"
        old_dir.mkdir()
        mgr = WorkspaceManager(workspace_dir=tmp_path / "workspace")
        result = mgr.migrate_soul_md(old_soul_dir=old_dir)
        assert result is False

    def test_migrate_soul_md_already_symlink(self, tmp_path: Path) -> None:
        old_dir = tmp_path / "old"
        old_dir.mkdir()
        ws_dir = tmp_path / "workspace"
        ws_dir.mkdir()

        # Create a target and symlink
        target = ws_dir / "SOUL.md"
        target.write_text("target", encoding="utf-8")
        old_soul = old_dir / "SOUL.md"
        old_soul.symlink_to(target)

        mgr = WorkspaceManager(workspace_dir=ws_dir)
        result = mgr.migrate_soul_md(old_soul_dir=old_dir)
        assert result is False

    def test_migrate_soul_md_new_location_exists(self, tmp_path: Path) -> None:
        old_dir = tmp_path / "old"
        old_dir.mkdir()
        (old_dir / "SOUL.md").write_text("old content", encoding="utf-8")

        ws_dir = tmp_path / "workspace"
        ws_dir.mkdir()
        (ws_dir / "SOUL.md").write_text("new content", encoding="utf-8")

        mgr = WorkspaceManager(workspace_dir=ws_dir)
        result = mgr.migrate_soul_md(old_soul_dir=old_dir)
        assert result is False
        # Neither file was changed
        assert (ws_dir / "SOUL.md").read_text(encoding="utf-8") == "new content"
        assert (old_dir / "SOUL.md").read_text(encoding="utf-8") == "old content"


# ---------------------------------------------------------------------------
# Unit tests: load_all / read_file / write_file
# ---------------------------------------------------------------------------


class TestFileOperations:
    def test_load_all_empty_dir(self, tmp_path: Path) -> None:
        mgr = WorkspaceManager(workspace_dir=tmp_path)
        files = mgr.load_all()
        assert files == {}

    def test_load_all_with_files(self, tmp_path: Path) -> None:
        (tmp_path / "USER.md").write_text("user content", encoding="utf-8")
        (tmp_path / "MEMORY.md").write_text("memory content", encoding="utf-8")

        mgr = WorkspaceManager(workspace_dir=tmp_path)
        files = mgr.load_all()
        assert files["USER.md"] == "user content"
        assert files["MEMORY.md"] == "memory content"

    def test_load_all_nonexistent_dir(self, tmp_path: Path) -> None:
        mgr = WorkspaceManager(workspace_dir=tmp_path / "nonexistent")
        files = mgr.load_all()
        assert files == {}

    def test_load_all_includes_extra_md_files(self, tmp_path: Path) -> None:
        (tmp_path / "CUSTOM.md").write_text("custom", encoding="utf-8")
        mgr = WorkspaceManager(workspace_dir=tmp_path)
        files = mgr.load_all()
        assert "CUSTOM.md" in files

    def test_load_all_ignores_non_md_files(self, tmp_path: Path) -> None:
        (tmp_path / "notes.txt").write_text("not markdown", encoding="utf-8")
        (tmp_path / "USER.md").write_text("user", encoding="utf-8")
        mgr = WorkspaceManager(workspace_dir=tmp_path)
        files = mgr.load_all()
        assert "notes.txt" not in files
        assert "USER.md" in files

    def test_load_all_priority_order(self, tmp_path: Path) -> None:
        """Standard files are loaded in WORKSPACE_FILE_NAMES order."""
        for name in WORKSPACE_FILE_NAMES:
            (tmp_path / name).write_text(f"content of {name}", encoding="utf-8")
        mgr = WorkspaceManager(workspace_dir=tmp_path)
        files = mgr.load_all()
        keys = list(files.keys())
        for i, name in enumerate(WORKSPACE_FILE_NAMES):
            assert keys[i] == name

    def test_read_file_existing(self, tmp_path: Path) -> None:
        (tmp_path / "USER.md").write_text("hello", encoding="utf-8")
        mgr = WorkspaceManager(workspace_dir=tmp_path)
        content = mgr.read_file("USER.md")
        assert content == "hello"

    def test_read_file_nonexistent(self, tmp_path: Path) -> None:
        mgr = WorkspaceManager(workspace_dir=tmp_path)
        assert mgr.read_file("NOPE.md") is None

    def test_read_file_path_traversal_blocked(self, tmp_path: Path) -> None:
        secret = tmp_path.parent / "secret.md"
        secret.write_text("top secret", encoding="utf-8")
        mgr = WorkspaceManager(workspace_dir=tmp_path)
        assert mgr.read_file("../secret.md") is None

    def test_read_file_updates_cache(self, tmp_path: Path) -> None:
        (tmp_path / "USER.md").write_text("v1", encoding="utf-8")
        mgr = WorkspaceManager(workspace_dir=tmp_path)
        mgr.read_file("USER.md")
        assert mgr.files["USER.md"] == "v1"

    def test_write_file_new(self, tmp_path: Path) -> None:
        mgr = WorkspaceManager(workspace_dir=tmp_path)
        ok = mgr.write_file("NEW.md", "new content")
        assert ok is True
        assert (tmp_path / "NEW.md").read_text(encoding="utf-8") == "new content"

    def test_write_file_overwrite(self, tmp_path: Path) -> None:
        (tmp_path / "USER.md").write_text("old", encoding="utf-8")
        mgr = WorkspaceManager(workspace_dir=tmp_path)
        ok = mgr.write_file("USER.md", "new")
        assert ok is True
        assert (tmp_path / "USER.md").read_text(encoding="utf-8") == "new"

    def test_write_file_path_traversal_blocked(self, tmp_path: Path) -> None:
        mgr = WorkspaceManager(workspace_dir=tmp_path)
        ok = mgr.write_file("../evil.md", "bad")
        assert ok is False

    def test_write_file_slash_blocked(self, tmp_path: Path) -> None:
        mgr = WorkspaceManager(workspace_dir=tmp_path)
        ok = mgr.write_file("sub/file.md", "content")
        assert ok is False

    def test_write_file_creates_directory(self, tmp_path: Path) -> None:
        ws_dir = tmp_path / "new_workspace"
        mgr = WorkspaceManager(workspace_dir=ws_dir)
        ok = mgr.write_file("MEMORY.md", "memory")
        assert ok is True
        assert ws_dir.is_dir()

    def test_write_file_updates_cache(self, tmp_path: Path) -> None:
        mgr = WorkspaceManager(workspace_dir=tmp_path)
        mgr.write_file("USER.md", "cached")
        assert mgr.files["USER.md"] == "cached"


# ---------------------------------------------------------------------------
# Unit tests: list_files
# ---------------------------------------------------------------------------


class TestListFiles:
    def test_list_files_empty(self, tmp_path: Path) -> None:
        mgr = WorkspaceManager(workspace_dir=tmp_path)
        assert mgr.list_files() == []

    def test_list_files_with_content(self, tmp_path: Path) -> None:
        (tmp_path / "USER.md").write_text("user", encoding="utf-8")
        (tmp_path / "MEMORY.md").write_text("memory", encoding="utf-8")
        mgr = WorkspaceManager(workspace_dir=tmp_path)
        files = mgr.list_files()
        assert len(files) == 2
        filenames = {f["filename"] for f in files}
        assert filenames == {"USER.md", "MEMORY.md"}

    def test_list_files_metadata(self, tmp_path: Path) -> None:
        (tmp_path / "USER.md").write_text("hello world", encoding="utf-8")
        mgr = WorkspaceManager(workspace_dir=tmp_path)
        files = mgr.list_files()
        assert len(files) == 1
        entry = files[0]
        assert entry["filename"] == "USER.md"
        assert entry["size"] == 11  # len("hello world")
        assert isinstance(entry["modified"], float)

    def test_list_files_nonexistent_dir(self, tmp_path: Path) -> None:
        mgr = WorkspaceManager(workspace_dir=tmp_path / "nope")
        assert mgr.list_files() == []

    def test_list_files_sorted(self, tmp_path: Path) -> None:
        (tmp_path / "Z.md").write_text("z", encoding="utf-8")
        (tmp_path / "A.md").write_text("a", encoding="utf-8")
        mgr = WorkspaceManager(workspace_dir=tmp_path)
        files = mgr.list_files()
        filenames = [f["filename"] for f in files]
        assert filenames == ["A.md", "Z.md"]

    def test_list_files_ignores_non_md(self, tmp_path: Path) -> None:
        (tmp_path / "notes.txt").write_text("txt", encoding="utf-8")
        (tmp_path / "USER.md").write_text("md", encoding="utf-8")
        mgr = WorkspaceManager(workspace_dir=tmp_path)
        files = mgr.list_files()
        assert len(files) == 1
        assert files[0]["filename"] == "USER.md"


# ---------------------------------------------------------------------------
# Unit tests: context_for_prompt
# ---------------------------------------------------------------------------


class TestContextForPrompt:
    def test_context_for_prompt_excludes_empty(self, tmp_path: Path) -> None:
        (tmp_path / "USER.md").write_text("user content", encoding="utf-8")
        (tmp_path / "MEMORY.md").write_text("   \n  ", encoding="utf-8")
        mgr = WorkspaceManager(workspace_dir=tmp_path)
        mgr.load_all()
        ctx = mgr.context_for_prompt()
        assert "USER.md" in ctx
        assert "MEMORY.md" not in ctx

    def test_context_for_prompt_returns_copy(self, tmp_path: Path) -> None:
        (tmp_path / "USER.md").write_text("content", encoding="utf-8")
        mgr = WorkspaceManager(workspace_dir=tmp_path)
        mgr.load_all()
        ctx = mgr.context_for_prompt()
        ctx["USER.md"] = "modified"
        assert mgr.context_for_prompt()["USER.md"] == "content"

    def test_context_for_prompt_all_files(self, tmp_path: Path) -> None:
        for name in WORKSPACE_FILE_NAMES:
            (tmp_path / name).write_text(f"content of {name}", encoding="utf-8")
        mgr = WorkspaceManager(workspace_dir=tmp_path)
        mgr.load_all()
        ctx = mgr.context_for_prompt()
        for name in WORKSPACE_FILE_NAMES:
            assert name in ctx


# ---------------------------------------------------------------------------
# Unit tests: SystemPromptBuilder integration
# ---------------------------------------------------------------------------


class TestPromptBuilderIntegration:
    def test_workspace_files_injected_into_prompt(self, tmp_path: Path) -> None:
        from orbiter_claw.prompt_builder import SystemPromptBuilder

        ctx = {
            "USER.md": "# User Profile\nAlice, timezone UTC",
            "MEMORY.md": "# Memory\nUser prefers bullet points",
        }
        builder = SystemPromptBuilder(mode="full", context_files=ctx)
        prompt = builder.build()
        assert "USER.md" in prompt
        assert "Alice, timezone UTC" in prompt
        assert "MEMORY.md" in prompt
        assert "User prefers bullet points" in prompt

    def test_context_for_prompt_works_with_builder(self, tmp_path: Path) -> None:
        from orbiter_claw.prompt_builder import SystemPromptBuilder

        (tmp_path / "SOUL.md").write_text("# Personality\nHelpful\n", encoding="utf-8")
        (tmp_path / "USER.md").write_text("# User Profile\nBob\n", encoding="utf-8")

        mgr = WorkspaceManager(workspace_dir=tmp_path)
        mgr.load_all()
        ctx = mgr.context_for_prompt()

        builder = SystemPromptBuilder(mode="full", context_files=ctx)
        prompt = builder.build()
        assert "Helpful" in prompt
        assert "Bob" in prompt
        assert "embody its persona and tone" in prompt

    def test_empty_workspace_produces_no_context(self, tmp_path: Path) -> None:
        from orbiter_claw.prompt_builder import SystemPromptBuilder

        mgr = WorkspaceManager(workspace_dir=tmp_path)
        mgr.load_all()
        ctx = mgr.context_for_prompt()
        assert ctx == {}

        builder = SystemPromptBuilder(mode="full", context_files=ctx)
        prompt = builder.build()
        assert "Project Context" not in prompt


# ---------------------------------------------------------------------------
# Tests: Hot-reload via watchdog
# ---------------------------------------------------------------------------


class TestWorkspaceHotReload:
    def test_start_stop_watching(self, tmp_path: Path) -> None:
        mgr = WorkspaceManager(workspace_dir=tmp_path)
        mgr.start_watching()
        assert mgr._observer is not None
        mgr.stop_watching()
        assert mgr._observer is None

    def test_start_watching_idempotent(self, tmp_path: Path) -> None:
        mgr = WorkspaceManager(workspace_dir=tmp_path)
        mgr.start_watching()
        observer1 = mgr._observer
        mgr.start_watching()
        assert mgr._observer is observer1
        mgr.stop_watching()

    def test_stop_watching_when_not_started(self, tmp_path: Path) -> None:
        mgr = WorkspaceManager(workspace_dir=tmp_path)
        mgr.stop_watching()  # Should not raise

    def test_file_change_triggers_reload(self, tmp_path: Path) -> None:
        user_md = tmp_path / "USER.md"
        user_md.write_text("original", encoding="utf-8")

        mgr = WorkspaceManager(workspace_dir=tmp_path)
        mgr.load_all()
        assert mgr.files.get("USER.md") == "original"

        mgr.start_watching()
        try:
            user_md.write_text("updated", encoding="utf-8")
            time.sleep(1.5)
            assert mgr.files.get("USER.md") == "updated"
        finally:
            mgr.stop_watching()

    def test_handler_on_modified_md_file(self) -> None:
        called: list[bool] = []

        def callback() -> None:
            called.append(True)

        handler = _WorkspaceFileHandler(callback)
        event = MagicMock()
        event.is_directory = False
        event.src_path = "/some/path/USER.md"
        handler.on_modified(event)
        assert len(called) == 1

    def test_handler_ignores_non_md_files(self) -> None:
        called: list[bool] = []

        def callback() -> None:
            called.append(True)

        handler = _WorkspaceFileHandler(callback)
        event = MagicMock()
        event.is_directory = False
        event.src_path = "/some/path/config.toml"
        handler.on_modified(event)
        assert len(called) == 0

    def test_handler_ignores_directories(self) -> None:
        called: list[bool] = []

        def callback() -> None:
            called.append(True)

        handler = _WorkspaceFileHandler(callback)
        event = MagicMock()
        event.is_directory = True
        event.src_path = "/some/path/USER.md"
        handler.on_modified(event)
        assert len(called) == 0

    def test_handler_on_created_md_file(self) -> None:
        called: list[bool] = []

        def callback() -> None:
            called.append(True)

        handler = _WorkspaceFileHandler(callback)
        event = MagicMock()
        event.is_directory = False
        event.src_path = "/some/path/MEMORY.md"
        handler.on_created(event)
        assert len(called) == 1


# ---------------------------------------------------------------------------
# Tests: Module-level singleton
# ---------------------------------------------------------------------------


class TestModuleSingleton:
    def test_init_workspace_sets_singleton(self, tmp_path: Path) -> None:
        mgr = WorkspaceManager(workspace_dir=tmp_path)
        init_workspace(mgr)
        assert get_workspace() is mgr
        # Reset
        init_workspace(WorkspaceManager(workspace_dir=tmp_path))

    def test_get_workspace_creates_default(self) -> None:
        import orbiter_claw.workspace as mod

        old = mod._workspace
        mod._workspace = None
        try:
            mgr = get_workspace()
            assert mgr is not None
            assert isinstance(mgr, WorkspaceManager)
        finally:
            mod._workspace = old


# ---------------------------------------------------------------------------
# Tests: files property
# ---------------------------------------------------------------------------


class TestFilesProperty:
    def test_files_returns_copy(self, tmp_path: Path) -> None:
        (tmp_path / "USER.md").write_text("content", encoding="utf-8")
        mgr = WorkspaceManager(workspace_dir=tmp_path)
        mgr.load_all()
        files = mgr.files
        files["USER.md"] = "modified"
        assert mgr.files["USER.md"] == "content"

    def test_files_empty_initially(self, tmp_path: Path) -> None:
        mgr = WorkspaceManager(workspace_dir=tmp_path)
        assert mgr.files == {}


# ---------------------------------------------------------------------------
# Tests: default file contents
# ---------------------------------------------------------------------------


class TestDefaultContents:
    def test_all_default_files_have_content(self) -> None:
        for filename, content in DEFAULT_FILE_CONTENTS.items():
            assert len(content) > 0, f"{filename} has empty default content"
            assert filename.endswith(".md")

    def test_workspace_file_names_constant(self) -> None:
        expected = {"SOUL.md", "AGENTS.md", "USER.md", "IDENTITY.md", "TOOLS.md", "MEMORY.md"}
        assert set(WORKSPACE_FILE_NAMES) == expected

    def test_soul_md_not_in_default_file_contents(self) -> None:
        """SOUL.md is managed by PersonalityManager, not in DEFAULT_FILE_CONTENTS."""
        assert "SOUL.md" not in DEFAULT_FILE_CONTENTS


# ---------------------------------------------------------------------------
# API endpoint tests
# ---------------------------------------------------------------------------


class TestWorkspaceAPI:
    async def test_list_workspace_files(self, client: AsyncClient, tmp_path: Path) -> None:
        mgr = WorkspaceManager(workspace_dir=tmp_path)
        mgr.initialize()
        mgr.load_all()
        init_workspace(mgr)

        resp = await client.get("/api/v1/workspace")
        assert resp.status_code == 200
        data = resp.json()
        assert "files" in data
        assert isinstance(data["files"], list)
        assert len(data["files"]) > 0

    async def test_read_workspace_file(self, client: AsyncClient, tmp_path: Path) -> None:
        (tmp_path / "USER.md").write_text("test user content", encoding="utf-8")
        mgr = WorkspaceManager(workspace_dir=tmp_path)
        mgr.load_all()
        init_workspace(mgr)

        resp = await client.get("/api/v1/workspace/USER.md")
        assert resp.status_code == 200
        data = resp.json()
        assert data["filename"] == "USER.md"
        assert data["content"] == "test user content"

    async def test_read_workspace_file_not_found(self, client: AsyncClient, tmp_path: Path) -> None:
        mgr = WorkspaceManager(workspace_dir=tmp_path)
        init_workspace(mgr)

        resp = await client.get("/api/v1/workspace/NONEXISTENT.md")
        assert resp.status_code == 404

    async def test_update_workspace_file(self, client: AsyncClient, tmp_path: Path) -> None:
        mgr = WorkspaceManager(workspace_dir=tmp_path)
        mgr.initialize()
        mgr.load_all()
        init_workspace(mgr)

        new_content = "# User Profile\nUpdated content\n"
        resp = await client.put(
            "/api/v1/workspace/USER.md",
            json={"content": new_content},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["filename"] == "USER.md"
        assert data["content"] == new_content

        # Verify persisted to disk
        assert (tmp_path / "USER.md").read_text(encoding="utf-8") == new_content

    async def test_update_workspace_file_non_md_rejected(
        self, client: AsyncClient, tmp_path: Path
    ) -> None:
        mgr = WorkspaceManager(workspace_dir=tmp_path)
        init_workspace(mgr)

        resp = await client.put(
            "/api/v1/workspace/config.toml",
            json={"content": "evil"},
        )
        assert resp.status_code == 400

    async def test_update_workspace_file_creates_new(
        self, client: AsyncClient, tmp_path: Path
    ) -> None:
        mgr = WorkspaceManager(workspace_dir=tmp_path)
        mgr.initialize()
        init_workspace(mgr)

        resp = await client.put(
            "/api/v1/workspace/CUSTOM.md",
            json={"content": "# Custom\nNew file\n"},
        )
        assert resp.status_code == 200
        assert (tmp_path / "CUSTOM.md").exists()

    async def test_list_after_update(self, client: AsyncClient, tmp_path: Path) -> None:
        mgr = WorkspaceManager(workspace_dir=tmp_path)
        init_workspace(mgr)

        # Create a file via API
        await client.put(
            "/api/v1/workspace/TEST.md",
            json={"content": "test content"},
        )

        # List should include it
        resp = await client.get("/api/v1/workspace")
        data = resp.json()
        filenames = {f["filename"] for f in data["files"]}
        assert "TEST.md" in filenames
