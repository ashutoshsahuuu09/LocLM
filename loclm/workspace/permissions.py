"""Workspace Permission Module — LocLM V9.

Defines granular workspace permission levels (READ, WRITE, FULL, DENY)
and permission validation helpers.
"""

from __future__ import annotations

import enum
from typing import Any


class WorkspacePermission(str, enum.Enum):
    """Permission level granted to LocLM for a specific workspace directory."""

    READ = "read"      # Can list, view, search, and inspect files. No edits.
    WRITE = "write"    # Can create, modify, rename files. Destructive ops require confirm.
    FULL = "full"      # Full workspace operation rights. Dangerous ops confirm.
    DENY = "deny"      # Explicitly denied access.

    @property
    def can_read(self) -> bool:
        """Check if permission allows reading files."""
        return self in (WorkspacePermission.READ, WorkspacePermission.WRITE, WorkspacePermission.FULL)

    @property
    def can_write(self) -> bool:
        """Check if permission allows creating/modifying files."""
        return self in (WorkspacePermission.WRITE, WorkspacePermission.FULL)

    @property
    def can_modify_structure(self) -> bool:
        """Check if permission allows structural changes."""
        return self in (WorkspacePermission.WRITE, WorkspacePermission.FULL)

    def is_sufficient_for(self, required: WorkspacePermission) -> bool:
        """Check if current permission meets or exceeds required permission."""
        if self == WorkspacePermission.DENY:
            return False
        if required == WorkspacePermission.READ:
            return self.can_read
        if required == WorkspacePermission.WRITE:
            return self.can_write
        if required == WorkspacePermission.FULL:
            return self == WorkspacePermission.FULL
        return False
