"""Terminal execution tools for LocLM (V2).

Provides safe command execution in the local terminal shell:
- RunTerminalCommandTool
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any

from loclm.tools.base import BaseTool, ToolCategory, ToolParameter, ToolPermissionLevel, ToolResult
from loclm.tools.security import SecurityGuard

logger = logging.getLogger(__name__)


class RunTerminalCommandTool(BaseTool):
    """Tool to execute a shell command in the local terminal safely."""

    name = "run_terminal_command"
    description = "Execute a shell command locally in the terminal and capture its output."
    category = ToolCategory.TERMINAL
    permission_level = ToolPermissionLevel.CONFIRM

    def __init__(self, guard: SecurityGuard | None = None) -> None:
        self._guard = guard or SecurityGuard()

    def get_parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="command",
                type="string",
                description="The terminal command line to run",
                required=True,
            ),
            ToolParameter(
                name="cwd",
                type="string",
                description="Working directory for the command (default: current directory)",
                required=False,
                default=".",
            ),
            ToolParameter(
                name="timeout",
                type="integer",
                description="Timeout limit in seconds (default: 30)",
                required=False,
                default=30,
            ),
        ]

    async def execute(self, **kwargs: Any) -> ToolResult:
        command = kwargs.get("command", "")
        cwd_str = kwargs.get("cwd", ".") or "."
        timeout = kwargs.get("timeout", 30) or 30

        if not command:
            return ToolResult(success=False, error="Parameter 'command' is required")

        # Security check
        is_safe, reason = self._guard.validate_command(command)
        if not is_safe:
            return ToolResult(success=False, error=f"Command blocked by security guard: {reason}")

        cwd_path = Path(cwd_str).resolve()
        is_valid_path, path_reason = self._guard.validate_path(cwd_path)
        if not is_valid_path:
            return ToolResult(success=False, error=f"Invalid working directory: {path_reason}")

        try:
            # Run command asynchronously in shell
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(cwd_path),
            )

            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(), timeout=float(timeout)
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                return ToolResult(
                    success=False,
                    error=f"Command execution timed out after {timeout} seconds.",
                    metadata={"command": command, "timed_out": True},
                )

            stdout_str = stdout_bytes.decode("utf-8", errors="replace")
            stderr_str = stderr_bytes.decode("utf-8", errors="replace")

            output = stdout_str
            if stderr_str:
                output = f"{stdout_str}\n--- STDERR ---\n{stderr_str}" if stdout_str else stderr_str

            # Truncate output if too long (>100KB)
            if len(output) > 100_000:
                output = output[:100_000] + "\n... [Output truncated at 100KB limit]"

            success = (proc.returncode == 0)

            return ToolResult(
                success=success,
                output=output,
                error=stderr_str if not success else None,
                metadata={
                    "command": command,
                    "exit_code": proc.returncode,
                    "cwd": str(cwd_path),
                },
            )

        except Exception as e:
            logger.error("Error running terminal command: %s", e)
            return ToolResult(success=False, error=f"Failed to execute command: {e}")
