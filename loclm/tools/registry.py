"""Central Tool Registry for LocLM (V2).

Manages tool discovery, registration, security checks, and execution routing.
"""

from __future__ import annotations

import logging
from typing import Any

from loclm.tools.base import BaseTool, ToolCategory, ToolPermissionLevel, ToolResult
from loclm.tools.filesystem import FileInfoTool, ListDirTool, ReadFileTool, SearchFilesTool, WriteFileTool
from loclm.tools.git_tools import GitBranchTool, GitCommitTool, GitDiffTool, GitLogTool, GitStatusTool
from loclm.tools.github import GitHubRemoteTool, GitHubStatusTool
from loclm.tools.python_exec import RunPythonScriptTool
from loclm.tools.security import SecurityGuard
from loclm.tools.terminal import RunTerminalCommandTool

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Registry and dispatcher for all registered local tools."""

    def __init__(self, guard: SecurityGuard | None = None) -> None:
        self._guard = guard or SecurityGuard()
        self._tools: dict[str, BaseTool] = {}
        self._register_default_tools()

    def _register_default_tools(self) -> None:
        """Register all core V2 local tools by default."""
        default_tools: list[BaseTool] = [
            # Filesystem
            ReadFileTool(self._guard),
            WriteFileTool(self._guard),
            ListDirTool(self._guard),
            SearchFilesTool(self._guard),
            FileInfoTool(self._guard),
            # Terminal
            RunTerminalCommandTool(self._guard),
            # Python
            RunPythonScriptTool(self._guard),
            # Git & GitHub
            GitStatusTool(self._guard),
            GitDiffTool(self._guard),
            GitLogTool(self._guard),
            GitCommitTool(self._guard),
            GitBranchTool(self._guard),
            GitHubStatusTool(self._guard),
            GitHubRemoteTool(self._guard),
        ]

        for t in default_tools:
            self.register_tool(t)

    def register_tool(self, tool: BaseTool) -> None:
        """Register a new tool instance in the registry."""
        if tool.name in self._tools:
            logger.warning("Overwriting registered tool: %s", tool.name)
        self._tools[tool.name] = tool
        logger.debug("Registered tool: %s (%s)", tool.name, tool.category.value)

    def get_tool(self, name: str) -> BaseTool | None:
        """Retrieve a registered tool by name."""
        return self._tools.get(name)

    def list_tools(self, category: ToolCategory | None = None) -> list[BaseTool]:
        """List registered tools, optionally filtered by category."""
        if category is None:
            return list(self._tools.values())
        return [t for t in self._tools.values() if t.category == category]

    def export_tools_schema(self) -> list[dict[str, Any]]:
        """Export all tools as JSON Schema function specs for LLM calling."""
        return [t.to_dict() for t in self._tools.values()]

    async def execute_tool(self, tool_name: str, **kwargs: Any) -> ToolResult:
        """Execute a tool by name with security checks.

        Args:
            tool_name: Name of the registered tool.
            **kwargs: Arguments to pass to the tool.

        Returns:
            ToolResult of the execution.
        """
        tool = self.get_tool(tool_name)
        if not tool:
            return ToolResult(
                success=False,
                error=f"Tool '{tool_name}' is not registered in the tool registry.",
            )

        # Check permissions
        perm = self._guard.check_tool_permission(tool)
        if perm == ToolPermissionLevel.BLOCK:
            return ToolResult(
                success=False,
                error=f"Execution of tool '{tool_name}' is blocked by policy.",
            )

        logger.info("Executing tool '%s' (category=%s)", tool_name, tool.category.value)
        try:
            result = await tool.execute(**kwargs)
            return result
        except Exception as e:
            logger.error("Exception during tool '%s' execution: %s", tool_name, e)
            return ToolResult(
                success=False,
                error=f"Tool execution error: {e}",
            )
