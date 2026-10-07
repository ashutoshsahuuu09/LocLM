"""Workspace Context Module — LocLM V9.

Maintains active local workspace state: current workspace, directory path,
active permission level, active agent role, and task state.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from loclm.workspace.permissions import WorkspacePermission
from loclm.workspace.registry import WorkspaceEntry

logger = logging.getLogger(__name__)


class WorkspaceContext(BaseModel):
    """Active runtime context for local workspace operations."""

    active_workspace: WorkspaceEntry | None = Field(default=None)
    current_directory: Path = Field(default_factory=Path.cwd)
    active_agent_role: str = Field(default="general")
    current_task: str = Field(default="")
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def permission(self) -> WorkspacePermission:
        """Current effective permission level."""
        if self.active_workspace:
            return self.active_workspace.permission
        return WorkspacePermission.DENY

    @property
    def workspace_name(self) -> str:
        """Active workspace name or 'None'."""
        return self.active_workspace.name if self.active_workspace else "None"

    def set_active_workspace(self, entry: WorkspaceEntry) -> None:
        """Set active workspace context."""
        self.active_workspace = entry
        self.current_directory = entry.resolved_path
        logger.info("Active workspace switched to '%s' [%s]", entry.name, entry.path)

    def get_summary(self) -> dict[str, str]:
        """Return workspace context status dict for display/CLI."""
        return {
            "workspace": self.workspace_name,
            "path": str(self.current_directory),
            "permission": self.permission.value.upper(),
            "active_agent": self.active_agent_role,
            "task": self.current_task or "None",
        }
