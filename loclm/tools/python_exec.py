"""Isolated Python execution engine for LocLM (V2).

Provides isolated Python code execution in a sub-process:
- RunPythonScriptTool
"""

from __future__ import annotations

import asyncio
import logging
import sys
import tempfile
from pathlib import Path
from typing import Any

from loclm.tools.base import BaseTool, ToolCategory, ToolParameter, ToolPermissionLevel, ToolResult
from loclm.tools.security import SecurityGuard

logger = logging.getLogger(__name__)


class RunPythonScriptTool(BaseTool):
    """Tool to execute Python code in an isolated sub-process."""

    name = "run_python_script"
    description = "Run a Python code snippet in an isolated subprocess and return stdout/stderr."
    category = ToolCategory.PYTHON
    permission_level = ToolPermissionLevel.CONFIRM

    def __init__(self, guard: SecurityGuard | None = None) -> None:
        self._guard = guard or SecurityGuard()

    def get_parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="code",
                type="string",
                description="Python code string to execute",
                required=True,
            ),
            ToolParameter(
                name="timeout",
                type="integer",
                description="Execution timeout in seconds (default: 30)",
                required=False,
                default=30,
            ),
        ]

    async def execute(self, **kwargs: Any) -> ToolResult:
        code = kwargs.get("code", "")
        timeout = kwargs.get("timeout", 30) or 30

        if not code.strip():
            return ToolResult(success=False, error="Parameter 'code' cannot be empty")

        # Create temporary script file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as temp_file:
            temp_file.write(code)
            temp_path = Path(temp_file.name)

        try:
            cmd = f'"{sys.executable}" "{temp_path}"'
            proc = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
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
                    error=f"Python script execution timed out after {timeout} seconds.",
                    metadata={"timed_out": True},
                )

            stdout_str = stdout_bytes.decode("utf-8", errors="replace")
            stderr_str = stderr_bytes.decode("utf-8", errors="replace")

            output = stdout_str
            if stderr_str:
                output = f"{stdout_str}\n--- STDERR ---\n{stderr_str}" if stdout_str else stderr_str

            return ToolResult(
                success=(proc.returncode == 0),
                output=output,
                error=stderr_str if proc.returncode != 0 else None,
                metadata={"exit_code": proc.returncode},
            )

        except Exception as e:
            logger.error("Python script execution error: %s", e)
            return ToolResult(success=False, error=f"Failed to execute Python script: {e}")
        finally:
            # Clean up temp file
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except Exception:
                    pass
