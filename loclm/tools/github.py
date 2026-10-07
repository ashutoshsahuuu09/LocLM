"""GitHub integration tools for LocLM.

Provides GitHub account status, repo operations, and remote linking:
- GitHubStatusTool
- GitHubRemoteTool
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from loclm.tools.base import BaseTool, ToolCategory, ToolParameter, ToolPermissionLevel, ToolResult
from loclm.tools.security import SecurityGuard

logger = logging.getLogger(__name__)


class GitHubStatusTool(BaseTool):
    """Tool to check local Git identity and GitHub connection status."""

    name = "github_status"
    description = "Check local Git user identity, configured remotes, and GitHub status."
    category = ToolCategory.GIT
    permission_level = ToolPermissionLevel.ALLOW

    def __init__(self, guard: SecurityGuard | None = None) -> None:
        self._guard = guard or SecurityGuard()

    def get_parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="cwd",
                type="string",
                description="Target repository directory (default: current directory)",
                required=False,
                default=".",
            )
        ]

    async def execute(self, **kwargs: Any) -> ToolResult:
        cwd_path = Path(kwargs.get("cwd", ".") or ".").resolve()

        # Check git user config
        proc_user = await asyncio.create_subprocess_shell(
            "git config user.name", stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, cwd=str(cwd_path)
        )
        user_out, _ = await proc_user.communicate()
        user_name = user_out.decode("utf-8", errors="replace").strip()

        proc_email = await asyncio.create_subprocess_shell(
            "git config user.email", stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, cwd=str(cwd_path)
        )
        email_out, _ = await proc_email.communicate()
        user_email = email_out.decode("utf-8", errors="replace").strip()

        # Check remote origin
        proc_remote = await asyncio.create_subprocess_shell(
            "git remote -v", stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, cwd=str(cwd_path)
        )
        remote_out, _ = await proc_remote.communicate()
        remotes = remote_out.decode("utf-8", errors="replace").strip()

        info = (
            f"Git User Name:  {user_name or 'Not set'}\n"
            f"Git User Email: {user_email or 'Not set'}\n"
            f"Configured Remotes:\n{remotes if remotes else 'No remotes configured.'}"
        )
        return ToolResult(
            success=True,
            output=info,
            metadata={"user_name": user_name, "user_email": user_email, "has_remotes": bool(remotes)},
        )


class GitHubRemoteTool(BaseTool):
    """Tool to link local repository to a GitHub remote URL."""

    name = "github_set_remote"
    description = "Add or update the GitHub remote origin URL for the repository."
    category = ToolCategory.GIT
    permission_level = ToolPermissionLevel.CONFIRM

    def __init__(self, guard: SecurityGuard | None = None) -> None:
        self._guard = guard or SecurityGuard()

    def get_parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="url",
                type="string",
                description="GitHub repository URL (e.g., https://github.com/username/LocLM.git)",
                required=True,
            ),
            ToolParameter(
                name="remote_name",
                type="string",
                description="Remote name (default: origin)",
                required=False,
                default="origin",
            ),
            ToolParameter(
                name="cwd",
                type="string",
                description="Target directory",
                required=False,
                default=".",
            ),
        ]

    async def execute(self, **kwargs: Any) -> ToolResult:
        url = kwargs.get("url", "").strip()
        remote_name = kwargs.get("remote_name", "origin") or "origin"
        cwd_path = Path(kwargs.get("cwd", ".") or ".").resolve()

        if not url:
            return ToolResult(success=False, error="Parameter 'url' is required.")

        # Check if remote already exists
        proc_check = await asyncio.create_subprocess_shell(
            f"git remote get-url {remote_name}", stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, cwd=str(cwd_path)
        )
        _, stderr_check = await proc_check.communicate()

        if proc_check.returncode == 0:
            cmd = f"git remote set-url {remote_name} {url}"
        else:
            cmd = f"git remote add {remote_name} {url}"

        proc = await asyncio.create_subprocess_shell(
            cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, cwd=str(cwd_path)
        )
        out, err = await proc.communicate()

        if proc.returncode == 0:
            return ToolResult(
                success=True,
                output=f"Successfully set Git remote '{remote_name}' to '{url}'",
            )
        else:
            return ToolResult(success=False, error=err.decode("utf-8", errors="replace"))


class GitHubAuthTool(BaseTool):
    """Tool to inspect and execute GitHub authentication via GitHub CLI (gh) or PAT."""

    name = "github_auth"
    description = "Check GitHub CLI authentication status or log in with a Personal Access Token (PAT)."
    category = ToolCategory.GIT
    permission_level = ToolPermissionLevel.ALLOW

    def __init__(self, guard: SecurityGuard | None = None) -> None:
        self._guard = guard or SecurityGuard()

    def get_parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="action",
                type="string",
                description="Action: 'status' (check auth status) or 'login_token' (login with token)",
                required=False,
                default="status",
            ),
            ToolParameter(
                name="token",
                type="string",
                description="GitHub Personal Access Token (required if action='login_token')",
                required=False,
                default="",
            ),
        ]

    async def execute(self, **kwargs: Any) -> ToolResult:
        action = kwargs.get("action", "status") or "status"
        token = kwargs.get("token", "").strip()

        if action == "login_token":
            if not token:
                return ToolResult(
                    success=False,
                    error="Personal Access Token ('token') is required for action 'login_token'.",
                )
            
            proc = await asyncio.create_subprocess_shell(
                f"echo {token} | gh auth login --with-token",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            out, err = await proc.communicate()
            if proc.returncode == 0:
                return ToolResult(
                    success=True,
                    output="Successfully authenticated GitHub CLI with Personal Access Token.",
                )
            else:
                return ToolResult(
                    success=False,
                    error=f"GitHub CLI token login failed: {err.decode('utf-8', errors='replace')}",
                )

        # Default action: check status
        proc = await asyncio.create_subprocess_shell(
            "gh auth status",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        out, err = await proc.communicate()
        status_text = (out.decode("utf-8", errors="replace") + "\n" + err.decode("utf-8", errors="replace")).strip()

        if proc.returncode == 0:
            return ToolResult(
                success=True,
                output=f"GitHub Authentication Status:\n{status_text}",
                metadata={"is_authenticated": True},
            )
        else:
            return ToolResult(
                success=False,
                output=f"GitHub Authentication Status: Not authenticated.\nDetails:\n{status_text}",
                metadata={"is_authenticated": False},
                error="GitHub CLI is not logged in. Run 'gh auth login' or provide a Personal Access Token.",
            )

