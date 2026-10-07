"""Git tools for LocLM (V2).

Provides local Git operations:
- GitStatusTool
- GitDiffTool
- GitLogTool
- GitCommitTool
- GitBranchTool
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from loclm.tools.base import BaseTool, ToolCategory, ToolParameter, ToolPermissionLevel, ToolResult
from loclm.tools.security import SecurityGuard

logger = logging.getLogger(__name__)


async def _run_git_cmd(args: list[str], cwd: Path) -> tuple[int, str, str]:
    """Helper to run a git sub-command safely."""
    cmd_str = f"git {' '.join(args)}"
    proc = await asyncio.create_subprocess_shell(
        cmd_str,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=str(cwd),
    )
    stdout, stderr = await proc.communicate()
    return (
        proc.returncode or 0,
        stdout.decode("utf-8", errors="replace"),
        stderr.decode("utf-8", errors="replace"),
    )


class GitStatusTool(BaseTool):
    """Tool to inspect Git repository working tree status."""

    name = "git_status"
    description = "Show the working tree status of the local Git repository."
    category = ToolCategory.GIT
    permission_level = ToolPermissionLevel.ALLOW

    def __init__(self, guard: SecurityGuard | None = None) -> None:
        self._guard = guard or SecurityGuard()

    def get_parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="cwd",
                type="string",
                description="Repository directory (default: current directory)",
                required=False,
                default=".",
            )
        ]

    async def execute(self, **kwargs: Any) -> ToolResult:
        cwd_path = Path(kwargs.get("cwd", ".") or ".").resolve()
        code, stdout, stderr = await _run_git_cmd(["status", "--short", "--branch"], cwd_path)

        if code != 0:
            return ToolResult(success=False, error=stderr or "Git status failed (is this a Git repo?)")

        return ToolResult(success=True, output=stdout or "Working tree clean.")


class GitDiffTool(BaseTool):
    """Tool to view uncommitted Git changes."""

    name = "git_diff"
    description = "Show changes in the working tree or staged commits."
    category = ToolCategory.GIT
    permission_level = ToolPermissionLevel.ALLOW

    def __init__(self, guard: SecurityGuard | None = None) -> None:
        self._guard = guard or SecurityGuard()

    def get_parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="staged",
                type="boolean",
                description="Show staged changes (--cached) if True",
                required=False,
                default=False,
            ),
            ToolParameter(
                name="cwd",
                type="string",
                description="Repository directory",
                required=False,
                default=".",
            ),
        ]

    async def execute(self, **kwargs: Any) -> ToolResult:
        staged = kwargs.get("staged", False)
        cwd_path = Path(kwargs.get("cwd", ".") or ".").resolve()

        args = ["diff"]
        if staged:
            args.append("--staged")

        code, stdout, stderr = await _run_git_cmd(args, cwd_path)
        if code != 0:
            return ToolResult(success=False, error=stderr or "Git diff failed.")

        return ToolResult(success=True, output=stdout or "(No diff changes)")


class GitLogTool(BaseTool):
    """Tool to view commit history."""

    name = "git_log"
    description = "Show recent commit history of the repository."
    category = ToolCategory.GIT
    permission_level = ToolPermissionLevel.ALLOW

    def __init__(self, guard: SecurityGuard | None = None) -> None:
        self._guard = guard or SecurityGuard()

    def get_parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="max_count",
                type="integer",
                description="Maximum number of commits to show (default: 10)",
                required=False,
                default=10,
            ),
            ToolParameter(
                name="cwd",
                type="string",
                description="Repository directory",
                required=False,
                default=".",
            ),
        ]

    async def execute(self, **kwargs: Any) -> ToolResult:
        count = kwargs.get("max_count", 10) or 10
        cwd_path = Path(kwargs.get("cwd", ".") or ".").resolve()

        args = ["log", f"-n{count}", "--oneline"]
        code, stdout, stderr = await _run_git_cmd(args, cwd_path)

        if code != 0:
            return ToolResult(success=False, error=stderr or "Git log failed.")

        return ToolResult(success=True, output=stdout or "(No commits yet)")


class GitCommitTool(BaseTool):
    """Tool to create a Git commit."""

    name = "git_commit"
    description = "Create a Git commit with a commit message."
    category = ToolCategory.GIT
    permission_level = ToolPermissionLevel.CONFIRM

    def __init__(self, guard: SecurityGuard | None = None) -> None:
        self._guard = guard or SecurityGuard()

    def get_parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="message",
                type="string",
                description="Commit message",
                required=True,
            ),
            ToolParameter(
                name="cwd",
                type="string",
                description="Repository directory",
                required=False,
                default=".",
            ),
        ]

    async def execute(self, **kwargs: Any) -> ToolResult:
        msg = kwargs.get("message", "")
        if not msg:
            return ToolResult(success=False, error="Commit message is required.")

        cwd_path = Path(kwargs.get("cwd", ".") or ".").resolve()
        code, stdout, stderr = await _run_git_cmd(["commit", "-m", f'"{msg}"'], cwd_path)

        if code != 0:
            return ToolResult(success=False, error=stderr or "Git commit failed.")

        return ToolResult(success=True, output=stdout)


class GitBranchTool(BaseTool):
    """Tool to list or view Git branches."""

    name = "git_branch"
    description = "List all branches in the local Git repository."
    category = ToolCategory.GIT
    permission_level = ToolPermissionLevel.ALLOW

    def __init__(self, guard: SecurityGuard | None = None) -> None:
        self._guard = guard or SecurityGuard()

    def get_parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="cwd",
                type="string",
                description="Repository directory",
                required=False,
                default=".",
            )
        ]

    async def execute(self, **kwargs: Any) -> ToolResult:
        cwd_path = Path(kwargs.get("cwd", ".") or ".").resolve()
        code, stdout, stderr = await _run_git_cmd(["branch", "-a"], cwd_path)

        if code != 0:
            return ToolResult(success=False, error=stderr or "Git branch failed.")

        return ToolResult(success=True, output=stdout)
