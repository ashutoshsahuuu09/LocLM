"""Path Guard Module — LocLM V9.

Strict security guard enforcing workspace path boundary isolation
and blocking path traversal exploits (e.g. '../').
"""

from __future__ import annotations

import logging
from pathlib import Path

from loclm.security.guard import PolicyViolationError

logger = logging.getLogger(__name__)


class PathGuard:
    """Audits file paths to ensure operations remain strictly within approved workspace boundaries."""

    @staticmethod
    def validate_workspace_path(
        target_path: str | Path,
        workspace_root: str | Path,
    ) -> Path:
        """Validate target_path is inside workspace_root and has no path traversal.

        Args:
            target_path: Target file/directory path.
            workspace_root: Approved workspace root path.

        Returns:
            Resolved target Path.

        Raises:
            PolicyViolationError: If target_path escapes workspace_root.
        """
        resolved_root = Path(workspace_root).resolve()
        resolved_target = Path(target_path).resolve()

        try:
            resolved_target.relative_to(resolved_root)
        except ValueError:
            logger.error("[SECURITY VIOLATION] Path traversal attempt: '%s' escapes root '%s'", target_path, resolved_root)
            raise PolicyViolationError(
                f"ACCESS DENIED: Path traversal detected. '{target_path}' escapes workspace boundary '{resolved_root}'."
            )

        return resolved_target
