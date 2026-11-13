"""System prompt builder for orbiter-claw agent.

Assembles multi-section system prompts from workspace files, tool descriptions,
skill lists, runtime info, and safety rules. Mirrors OpenClaw's prompt architecture.
"""

from __future__ import annotations

import platform
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

PromptMode = Literal["full", "minimal", "none"]

SILENT_REPLY_TOKEN = "<<SILENT>>"
HEARTBEAT_OK_TOKEN = "HEARTBEAT_OK"

# Unicode control character pattern (C0/C1 controls except common whitespace)
_CONTROL_CHAR_RE = re.compile(
    r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]",
)

# Workspace files loaded in priority order
WORKSPACE_FILES = [
    "SOUL.md",
    "AGENTS.md",
    "USER.md",
    "IDENTITY.md",
    "TOOLS.md",
    "MEMORY.md",
]


def sanitize_prompt_content(text: str) -> str:
    """Strip Unicode control characters from text to prevent prompt injection.

    Preserves normal whitespace (\\n, \\r, \\t, space) while removing
    C0/C1 control characters that could be used for injection.
    """
    return _CONTROL_CHAR_RE.sub("", text)


class SystemPromptBuilder:
    """Builds structured system prompts for the orbiter-claw agent.

    Assembles sections in order:
    1. Identity
    2. Tooling
    3. Tool call style
    4. Safety rules
    5. Skills list
    6. Memory recall instructions
    7. Workspace info
    8. Current date/time with user timezone
    9. Workspace files (SOUL.md, AGENTS.md, etc. as Project Context)
    10. Messaging/channel context
    11. Runtime info (model, channel, capabilities)
    """

    def __init__(
        self,
        *,
        workspace_dir: str | Path | None = None,
        mode: PromptMode = "full",
        tool_names: list[str] | None = None,
        tool_summaries: dict[str, str] | None = None,
        skills_prompt: str | None = None,
        heartbeat_prompt: str | None = None,
        user_timezone: str | None = None,
        model: str | None = None,
        default_model: str | None = None,
        channel: str | None = None,
        capabilities: list[str] | None = None,
        extra_system_prompt: str | None = None,
        context_files: dict[str, str] | None = None,
    ) -> None:
        self.workspace_dir = Path(workspace_dir) if workspace_dir else None
        self.mode = mode
        self.tool_names = tool_names or []
        self.tool_summaries = tool_summaries or {}
        self.skills_prompt = skills_prompt
        self.heartbeat_prompt = heartbeat_prompt
        self.user_timezone = user_timezone
        self.model = model
        self.default_model = default_model
        self.channel = channel
        self.capabilities = capabilities or []
        self.extra_system_prompt = extra_system_prompt
        self._context_files = context_files

    def _is_minimal(self) -> bool:
        return self.mode in ("minimal", "none")

    def _build_identity(self) -> list[str]:
        return ["You are a personal AI assistant running inside orbiter-claw."]

    def _build_tooling(self) -> list[str]:
        lines = [
            "",
            "## Tooling",
            "Tool availability (filtered by policy):",
            "Tool names are case-sensitive. Call tools exactly as listed.",
        ]
        if self.tool_names:
            for name in self.tool_names:
                summary = self.tool_summaries.get(name, "")
                if summary:
                    lines.append(f"- {name}: {summary}")
                else:
                    lines.append(f"- {name}")
        else:
            lines.append("(No tools currently registered)")
        lines.append("")
        return lines

    def _build_tool_call_style(self) -> list[str]:
        if self._is_minimal():
            return []
        return [
            "## Tool Call Style",
            "Default: do not narrate routine, low-risk tool calls — just call the tool.",
            "Narrate only when it helps the user understand what you are doing: "
            "multi-step work, complex problems, sensitive actions (e.g., deletions), "
            "or when the user explicitly asks.",
            "Keep narration brief and value-dense; avoid repeating obvious steps.",
            "",
        ]

    def _build_safety(self) -> list[str]:
        if self._is_minimal():
            return []
        return [
            "## Safety",
            "You have no independent goals: do not pursue self-preservation, "
            "replication, resource acquisition, or power-seeking; avoid long-term "
            "plans beyond the user's request.",
            "Prioritize safety and human oversight over completion; if instructions "
            "conflict, pause and ask; comply with stop/pause/audit requests and "
            "never bypass safeguards.",
            "Do not manipulate or persuade anyone to expand access or disable "
            "safeguards. Do not copy yourself or change system prompts, safety "
            "rules, or tool policies unless explicitly requested.",
            "Do not exfiltrate private data. Prefer `trash` over `rm` "
            "(recoverable beats gone forever). When in doubt about external "
            "actions, ask first.",
            "",
        ]

    def _build_skills(self) -> list[str]:
        if self._is_minimal():
            return []
        trimmed = (self.skills_prompt or "").strip()
        if not trimmed:
            return []
        return [
            "## Skills",
            "Before replying: scan available skills.",
            "- If exactly one skill clearly applies: read its SKILL.md, then follow it.",
            "- If multiple could apply: choose the most specific one.",
            "- If none clearly apply: do not read any SKILL.md.",
            "",
            trimmed,
            "",
        ]

    def _build_memory_recall(self) -> list[str]:
        if self._is_minimal():
            return []
        return [
            "## Memory Recall",
            "Before answering anything about prior work, decisions, dates, people, "
            "preferences, or todos: search MEMORY.md + memory/*.md first. If low "
            "confidence after search, say you checked but found nothing relevant.",
            "",
        ]

    def _build_workspace_info(self) -> list[str]:
        ws = str(self.workspace_dir) if self.workspace_dir else "~/.orbiter-claw/workspace"
        lines = [
            "## Workspace",
            f"Your working directory is: {sanitize_prompt_content(ws)}",
            "Treat this directory as the single global workspace for file "
            "operations unless explicitly instructed otherwise.",
            "",
        ]
        return lines

    def _build_datetime(self) -> list[str]:
        if self._is_minimal():
            return []
        now = datetime.now(UTC)
        lines = [
            "## Current Date & Time",
            f"UTC: {now.strftime('%Y-%m-%d %H:%M:%S')}",
        ]
        if self.user_timezone:
            lines.append(f"User timezone: {sanitize_prompt_content(self.user_timezone)}")
        lines.append("")
        return lines

    def _load_workspace_files(self) -> list[tuple[str, str]]:
        """Load workspace files from disk, returning (filename, content) pairs."""
        if self._context_files is not None:
            return [
                (name, content) for name, content in self._context_files.items() if content.strip()
            ]

        if not self.workspace_dir or not self.workspace_dir.is_dir():
            return []

        files: list[tuple[str, str]] = []
        for filename in WORKSPACE_FILES:
            path = self.workspace_dir / filename
            if path.is_file():
                try:
                    content = path.read_text(encoding="utf-8")
                    if content.strip():
                        files.append((filename, content))
                except OSError:
                    continue
        return files

    def _build_project_context(self) -> list[str]:
        """Build project context section from workspace files."""
        ws_files = self._load_workspace_files()
        if not ws_files:
            return []

        has_soul = any(name.lower() == "soul.md" for name, _ in ws_files)

        lines = [
            "# Project Context",
            "",
            "The following workspace files have been loaded:",
        ]
        if has_soul:
            lines.append(
                "If SOUL.md is present, embody its persona and tone. "
                "Avoid stiff, generic replies; follow its guidance unless "
                "higher-priority instructions override it."
            )
        lines.append("")

        for filename, content in ws_files:
            sanitized = sanitize_prompt_content(content)
            lines.extend([f"## {filename}", "", sanitized, ""])

        return lines

    def _build_messaging(self) -> list[str]:
        if self._is_minimal():
            return []
        lines = [
            "## Messaging",
            "- Reply in current session automatically routes to the source channel.",
            "- `[System Message] ...` blocks are internal context and are not "
            "user-visible by default.",
        ]
        if self.channel:
            lines.append(f"- Current channel: {sanitize_prompt_content(self.channel)}")
        lines.append("")
        return lines

    def _build_silent_replies(self) -> list[str]:
        if self._is_minimal():
            return []
        return [
            "## Silent Replies",
            f"When you have nothing to say, respond with ONLY: {SILENT_REPLY_TOKEN}",
            "",
            "Rules:",
            "- It must be your ENTIRE message — nothing else.",
            f'- Never append it to an actual response (never include "{SILENT_REPLY_TOKEN}" '
            "in real replies).",
            "- Never wrap it in markdown or code blocks.",
            "",
        ]

    def _build_heartbeat(self) -> list[str]:
        if self._is_minimal():
            return []
        prompt_line = (
            f"Heartbeat prompt: {self.heartbeat_prompt}"
            if self.heartbeat_prompt
            else "Heartbeat prompt: (configured)"
        )
        return [
            "## Heartbeats",
            prompt_line,
            "If you receive a heartbeat poll and there is nothing that needs "
            "attention, reply exactly:",
            HEARTBEAT_OK_TOKEN,
            f'If something needs attention, do NOT include "{HEARTBEAT_OK_TOKEN}"; '
            "reply with the alert text instead.",
            "",
        ]

    def _build_runtime(self) -> list[str]:
        parts: list[str] = []
        parts.append(f"os={platform.system()} ({platform.machine()})")
        if self.model:
            parts.append(f"model={sanitize_prompt_content(self.model)}")
        if self.default_model:
            parts.append(f"default_model={sanitize_prompt_content(self.default_model)}")
        if self.channel:
            parts.append(f"channel={sanitize_prompt_content(self.channel)}")
            caps = ",".join(self.capabilities) if self.capabilities else "none"
            parts.append(f"capabilities={caps}")

        return [
            "## Runtime",
            f"Runtime: {' | '.join(parts)}",
        ]

    def build(self) -> str:
        """Assemble the complete system prompt.

        Returns the assembled prompt string with all sections appropriate
        for the configured prompt mode.
        """
        # "none" mode: just identity line
        if self.mode == "none":
            return "You are a personal AI assistant running inside orbiter-claw."

        lines: list[str] = []

        # 1. Identity
        lines.extend(self._build_identity())

        # 2. Tooling (always included for full and minimal)
        lines.extend(self._build_tooling())

        # 3. Tool call style (full only)
        lines.extend(self._build_tool_call_style())

        # 4. Safety (full only)
        lines.extend(self._build_safety())

        # 5. Skills (full only)
        lines.extend(self._build_skills())

        # 6. Memory recall (full only)
        lines.extend(self._build_memory_recall())

        # 7. Workspace info (always for full and minimal)
        lines.extend(self._build_workspace_info())

        # 8. Date/time (full only)
        lines.extend(self._build_datetime())

        # 9. Project context / workspace files (always for full and minimal)
        lines.extend(self._build_project_context())

        # 10. Messaging (full only)
        lines.extend(self._build_messaging())

        # Extra system prompt (group/subagent context)
        if self.extra_system_prompt:
            header = "## Subagent Context" if self.mode == "minimal" else "## Additional Context"
            lines.extend([header, self.extra_system_prompt, ""])

        # Silent replies (full only)
        lines.extend(self._build_silent_replies())

        # Heartbeats (full only)
        lines.extend(self._build_heartbeat())

        # 11. Runtime (always for full and minimal)
        lines.extend(self._build_runtime())

        return "\n".join(line for line in lines if line is not None)
