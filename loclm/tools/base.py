"""Base classes and schemas for LocLM tool system.

All tools (filesystem, terminal, python, git) inherit from BaseTool
and declare their parameters, category, and permission requirements.
"""

from __future__ import annotations

import abc
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class ToolCategory(StrEnum):
    """Category of local tools."""

    FILESYSTEM = "filesystem"
    TERMINAL = "terminal"
    PYTHON = "python"
    GIT = "git"
    SYSTEM = "system"


class ToolPermissionLevel(StrEnum):
    """Permission level required to execute a tool."""

    ALLOW = "allow"        # Automatic execution without confirmation
    CONFIRM = "confirm"    # Requires user approval before execution
    BLOCK = "block"        # Completely disabled by policy


class ToolParameter(BaseModel):
    """Definition of a tool input parameter."""

    name: str = Field(description="Parameter name")
    type: str = Field(default="string", description="JSON Schema parameter type (string, integer, boolean, object, array)")
    description: str = Field(description="Human-readable description of what this parameter does")
    required: bool = Field(default=True, description="Whether this parameter is mandatory")
    default: Any | None = Field(default=None, description="Default value if parameter is optional")


class ToolResult(BaseModel):
    """Standardized output returned by a tool execution."""

    success: bool = Field(description="True if execution succeeded, False if error occurred")
    output: str = Field(default="", description="Main text output / stdout")
    error: str | None = Field(default=None, description="Error message / stderr if failed")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Structured execution metadata")


class BaseTool(abc.ABC):
    """Abstract base class for all LocLM tools.

    Tools implement `execute(**kwargs)` and provide metadata for schema export.
    """

    name: str
    description: str
    category: ToolCategory
    permission_level: ToolPermissionLevel = ToolPermissionLevel.ALLOW

    @abc.abstractmethod
    def get_parameters(self) -> list[ToolParameter]:
        """Return parameter specification list for this tool."""
        ...

    @abc.abstractmethod
    async def execute(self, **kwargs: Any) -> ToolResult:
        """Execute tool logic asynchronously.

        Args:
            **kwargs: Arguments matching parameter specification.

        Returns:
            ToolResult containing status, output, and optional error.
        """
        ...

    def to_dict(self) -> dict[str, Any]:
        """Export tool spec as JSON schema dictionary for LLM function calling."""
        params = self.get_parameters()
        properties: dict[str, dict[str, Any]] = {}
        required: list[str] = []

        for p in params:
            properties[p.name] = {
                "type": p.type,
                "description": p.description,
            }
            if p.default is not None:
                properties[p.name]["default"] = p.default
            if p.required:
                required.append(p.name)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
            "category": self.category.value,
            "permission_level": self.permission_level.value,
        }
