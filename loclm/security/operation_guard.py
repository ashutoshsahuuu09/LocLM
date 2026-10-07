"""Operation Guard Module — LocLM V9.

Unified zero-trust operation auditor combining workspace boundary verification,
path traversal prevention, permission checking, and audit logging.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from loclm.security.guard import AuditRecord, PolicyGuard, PolicyViolationError, RiskLevel
from loclm.security.path_guard import PathGuard
from loclm.security.permission_guard import PermissionGuard
from loclm.workspace.permissions import WorkspacePermission
from loclm.workspace.registry import WorkspaceEntry

logger = logging.getLogger(__name__)


class OperationGuard:
    """Unified security auditor verifying every tool call and file operation against workspace policies."""

    def __init__(self, policy_guard: PolicyGuard | None = None) -> None:
        self.policy_guard = policy_guard or PolicyGuard()

    def audit_workspace_operation(
        self,
        workspace: WorkspaceEntry,
        target_path: str | Path,
        operation: str,
    ) -> bool:
        """Audit a file operation against an approved workspace entry.

        Args:
            workspace: The WorkspaceEntry context.
            target_path: Relative or absolute path to target file.
            operation: Operation type ('read', 'write', 'create', 'delete', 'modify').

        Returns:
            True if allowed.

        Raises:
            PolicyViolationError: If security policy or permission is violated.
        """
        # 1. Permission check
        PermissionGuard.enforce_permission(
            operation_type=operation,
            current_permission=workspace.permission,
            workspace_name=workspace.name,
        )

        # 2. Path traversal & boundary check
        PathGuard.validate_workspace_path(
            target_path=target_path,
            workspace_root=workspace.resolved_path,
        )

        # 3. Policy Guard audit logging
        risk = RiskLevel.MEDIUM if operation in ("write", "create", "modify", "delete") else RiskLevel.LOW
        record = AuditRecord(
            agent_role=workspace.name,
            action_type=f"workspace_{operation}",
            target_resource=str(target_path),
            allowed=True,
            risk_level=risk,
            details=f"Permitted operation '{operation}' in workspace '{workspace.name}'",
        )
        self.policy_guard.get_audit_trail().append(record)
        return True
