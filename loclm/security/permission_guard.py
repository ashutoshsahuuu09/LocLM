"""Permission Guard Module — LocLM V9.

Enforces workspace permission level constraints (READ, WRITE, FULL, DENY).
"""

from __future__ import annotations

import logging

from loclm.security.guard import PolicyViolationError
from loclm.workspace.permissions import WorkspacePermission

logger = logging.getLogger(__name__)


class PermissionGuard:
    """Audits requested file operations against workspace permission levels."""

    @staticmethod
    def enforce_permission(
        operation_type: str,
        current_permission: WorkspacePermission,
        workspace_name: str = "workspace",
    ) -> bool:
        """Enforce operation permissions.

        Args:
            operation_type: e.g. 'read', 'write', 'create', 'delete', 'modify'
            current_permission: Current WorkspacePermission level
            workspace_name: Friendly name for log error messages

        Returns:
            True if permitted.

        Raises:
            PolicyViolationError: If permission is insufficient.
        """
        op = operation_type.lower()

        if current_permission == WorkspacePermission.DENY:
            raise PolicyViolationError(f"ACCESS DENIED: Access to workspace '{workspace_name}' has been denied.")

        if op in ("read", "list", "search", "inspect"):
            if not current_permission.can_read:
                raise PolicyViolationError(
                    f"ACCESS DENIED: Read operation blocked for workspace '{workspace_name}'."
                )
            return True

        if op in ("write", "create", "modify", "rename", "delete", "mkdir"):
            if not current_permission.can_write:
                raise PolicyViolationError(
                    f"ACCESS DENIED: Workspace '{workspace_name}' has READ-ONLY permission. "
                    f"Operation '{op}' is blocked."
                )
            return True

        return True
