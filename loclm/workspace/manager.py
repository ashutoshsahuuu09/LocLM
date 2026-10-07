"""Workspace Manager Module — LocLM V9.

Central controller for workspace registration, directory routing, context tracking,
permission enforcement, and security boundary verification.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from loclm.workspace.context import WorkspaceContext
from loclm.workspace.permissions import WorkspacePermission
from loclm.workspace.registry import WorkspaceEntry, WorkspaceRegistry
from loclm.workspace.router import DirectoryRouter, WorkspaceMatch
from loclm.workspace.scanner import WorkspaceScanner

logger = logging.getLogger(__name__)


class WorkspaceManager:
    """Central manager for LocLM multi-directory workspaces."""

    def __init__(self, registry: WorkspaceRegistry | None = None) -> None:
        self.registry = registry or WorkspaceRegistry()
        self.router = DirectoryRouter()
        self.context = WorkspaceContext()
        self.scanner = WorkspaceScanner()

    def list_workspaces(self) -> list[WorkspaceEntry]:
        """Return all approved workspace entries."""
        return self.registry.list_workspaces()

    def add_workspace(
        self,
        name: str,
        path: str | Path,
        permission: WorkspacePermission | str = WorkspacePermission.WRITE,
    ) -> WorkspaceEntry:
        """Register a new approved workspace."""
        entry = self.registry.add_workspace(name, path, permission)
        if not self.context.active_workspace:
            self.context.set_active_workspace(entry)
        return entry

    def remove_workspace(self, name_or_path: str) -> bool:
        """Revoke workspace access permission. (Does NOT delete user files)."""
        removed = self.registry.remove_workspace(name_or_path)
        if removed and self.context.active_workspace and self.context.active_workspace.name == name_or_path:
            self.context.active_workspace = None
        return removed

    def use_workspace(self, name_or_path: str) -> WorkspaceEntry:
        """Switch active workspace context.

        Raises:
            KeyError: If workspace is not approved.
        """
        entry = self.registry.get_workspace(name_or_path)
        if not entry:
            raise KeyError(f"Workspace '{name_or_path}' is not registered or approved.")
        self.context.set_active_workspace(entry)
        return entry

    def set_permission(
        self,
        name_or_path: str,
        new_permission: WorkspacePermission | str,
    ) -> WorkspaceEntry:
        """Update permission level for a workspace."""
        entry = self.registry.update_permission(name_or_path, new_permission)
        if self.context.active_workspace and self.context.active_workspace.name == entry.name:
            self.context.active_workspace = entry
        return entry

    def route_task_to_workspace(self, user_prompt: str) -> WorkspaceMatch:
        """Determine which approved workspace matches the user prompt."""
        match = self.router.route_prompt(user_prompt, self.registry)
        if match.matched_workspace and not match.is_ambiguous:
            self.context.set_active_workspace(match.matched_workspace)
        return match

    def validate_file_operation(
        self,
        target_path: str | Path,
        required_permission: WorkspacePermission = WorkspacePermission.READ,
    ) -> tuple[bool, str, WorkspaceEntry | None]:
        """Validate if a file operation is permitted inside an approved workspace.

        Prevents path traversal ('../') outside approved workspace boundaries.

        Returns:
            Tuple of (is_allowed, reason_message, matched_workspace_entry).
        """
        resolved_target = Path(target_path).resolve()

        # Find containing workspace
        entry = self.registry.find_containing_workspace(resolved_target)
        if not entry:
            return (
                False,
                f"ACCESS DENIED: Path '{resolved_target}' is outside all approved workspaces.",
                None,
            )

        # Path traversal check: ensure resolved_target is inside workspace path
        ws_path = entry.resolved_path
        try:
            resolved_target.relative_to(ws_path)
        except ValueError:
            return (
                False,
                f"ACCESS DENIED: Path traversal detected outside workspace boundary '{ws_path}'.",
                entry,
            )

        # Check permission level
        if not entry.permission.is_sufficient_for(required_permission):
            return (
                False,
                f"ACCESS DENIED: Operation requires {required_permission.value.upper()} permission, "
                f"but workspace '{entry.name}' has {entry.permission.value.upper()} permission.",
                entry,
            )

        return True, "Access permitted.", entry

    def inspect_workspace(self, name_or_path: str | None = None) -> dict[str, Any]:
        """Inspect a workspace's file tree and project indicators."""
        target_entry = self.registry.get_workspace(name_or_path) if name_or_path else self.context.active_workspace
        if not target_entry:
            return {"error": "No workspace specified or active."}

        indicators = self.scanner.inspect_project_indicators(target_entry.resolved_path)
        indicators["permission"] = target_entry.permission.value.upper()
        indicators["approved"] = target_entry.approved
        return indicators
