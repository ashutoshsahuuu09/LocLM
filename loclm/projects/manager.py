"""Project Manager Module — LocLM V9.

Orchestrates project creation, template management, inspection, and verification
within approved workspaces.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from loclm.models.manager import ModelManager
from loclm.projects.creator import ProjectCreationAgent, ProjectPlan
from loclm.projects.inspector import ProjectInspector
from loclm.projects.templates import ProjectTemplate, TemplateRegistry
from loclm.projects.verifier import ProjectVerifier, VerificationReport
from loclm.workspace.manager import WorkspaceManager
from loclm.workspace.permissions import WorkspacePermission

logger = logging.getLogger(__name__)


class ProjectManager:
    """High-level manager for project operations within approved workspaces."""

    def __init__(
        self,
        workspace_manager: WorkspaceManager | None = None,
        model_manager: ModelManager | None = None,
    ) -> None:
        self.workspace_manager = workspace_manager or WorkspaceManager()
        self.model_manager = model_manager
        self.creator = ProjectCreationAgent(self.model_manager)
        self.inspector = ProjectInspector()
        self.verifier = ProjectVerifier()

    def list_templates(self) -> list[ProjectTemplate]:
        """List all available project templates."""
        return TemplateRegistry.list_templates()

    def create_project_plan(
        self,
        user_request: str,
        workspace_name_or_path: str | None = None,
    ) -> ProjectPlan:
        """Parse requirements into a ProjectPlan and verify target workspace permission."""
        ws_entry = (
            self.workspace_manager.registry.get_workspace(workspace_name_or_path)
            if workspace_name_or_path
            else self.workspace_manager.context.active_workspace
        )

        if not ws_entry:
            raise ValueError("No active or specified workspace found for project creation.")

        if not ws_entry.permission.can_write:
            raise PermissionError(
                f"Workspace '{ws_entry.name}' has {ws_entry.permission.value.upper()} permission. "
                f"Project creation requires WRITE or FULL permission."
            )

        return self.creator.parse_requirements_to_plan(user_request, ws_entry.path)

    def execute_project_creation(
        self,
        plan: ProjectPlan,
        overwrite_strategy: str = "keep",
    ) -> tuple[bool, list[str], VerificationReport]:
        """Execute a confirmed ProjectPlan."""
        # Security validation
        allowed, msg, _ = self.workspace_manager.validate_file_operation(
            plan.project_dir,
            required_permission=WorkspacePermission.WRITE,
        )

        if not allowed:
            raise PermissionError(msg)

        return self.creator.execute_plan(plan, overwrite_strategy=overwrite_strategy)

    def inspect_project(self, project_path_or_name: str) -> dict[str, Any]:
        """Inspect an existing project directory inside an approved workspace."""
        allowed, msg, entry = self.workspace_manager.validate_file_operation(
            project_path_or_name,
            required_permission=WorkspacePermission.READ,
        )

        if not allowed:
            raise PermissionError(msg)

        return self.inspector.inspect(project_path_or_name)

    def get_project_tree(self, project_path_or_name: str, max_depth: int = 3) -> str:
        """Generate directory tree for a project."""
        allowed, msg, _ = self.workspace_manager.validate_file_operation(
            project_path_or_name,
            required_permission=WorkspacePermission.READ,
        )

        if not allowed:
            raise PermissionError(msg)

        return self.workspace_manager.scanner.generate_tree(project_path_or_name, max_depth=max_depth)
