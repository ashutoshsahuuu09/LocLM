"""Comprehensive Pytest Suite for LocLM V9 — Multi-Directory Workspaces & Project Agent."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from loclm.agents.v9_orchestrator import V9AgentResponse, V9Orchestrator
from loclm.models.manager import ModelManager
from loclm.projects.creator import ProjectCreationAgent, ProjectPlan
from loclm.projects.inspector import ProjectInspector
from loclm.projects.manager import ProjectManager
from loclm.projects.templates import TemplateRegistry
from loclm.projects.verifier import ProjectVerifier
from loclm.security.guard import PolicyViolationError
from loclm.security.operation_guard import OperationGuard
from loclm.security.path_guard import PathGuard
from loclm.security.permission_guard import PermissionGuard
from loclm.workspace.context import WorkspaceContext
from loclm.workspace.manager import WorkspaceManager
from loclm.workspace.permissions import WorkspacePermission
from loclm.workspace.registry import WorkspaceEntry, WorkspaceRegistry
from loclm.workspace.router import DirectoryRouter


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def temp_workspace_dir(tmp_path):
    """Fixture providing temporary mock workspace directory structure."""
    ws1 = tmp_path / "Projects" / "CodeV"
    ws2 = tmp_path / "Projects" / "TradingBot"
    ws3 = tmp_path / "ML" / "MLProject"

    ws1.mkdir(parents=True, exist_ok=True)
    ws2.mkdir(parents=True, exist_ok=True)
    ws3.mkdir(parents=True, exist_ok=True)

    # Add sample files
    (ws1 / "main.py").write_text("print('CodeV')", encoding="utf-8")
    (ws2 / "bot.py").write_text("print('TradingBot')", encoding="utf-8")
    (ws3 / "train.py").write_text("print('ML')", encoding="utf-8")

    return tmp_path, ws1, ws2, ws3


@pytest.fixture
def workspace_manager(temp_workspace_dir):
    tmp_path, ws1, ws2, ws3 = temp_workspace_dir
    reg_file = tmp_path / "workspaces.json"
    registry = WorkspaceRegistry(registry_file=reg_file)

    registry.add_workspace("CodeV", ws1, WorkspacePermission.WRITE)
    registry.add_workspace("TradingBot", ws2, WorkspacePermission.WRITE)
    registry.add_workspace("MLProject", ws3, WorkspacePermission.READ)

    return WorkspaceManager(registry=registry)


# ---------------------------------------------------------------------------
# 1. Workspace Permission Tests (Acceptance Test 1 & 2)
# ---------------------------------------------------------------------------


def test_permission_unapproved_folder_blocked(workspace_manager, tmp_path):
    unapproved_dir = tmp_path / "UnapprovedFolder"
    unapproved_dir.mkdir(parents=True, exist_ok=True)
    target_file = unapproved_dir / "secret.txt"
    target_file.write_text("confidential", encoding="utf-8")

    allowed, msg, entry = workspace_manager.validate_file_operation(target_file, WorkspacePermission.READ)
    assert allowed is False
    assert "ACCESS DENIED" in msg
    assert entry is None


def test_permission_readonly_blocks_edits(workspace_manager):
    # MLProject has READ permission
    ml_entry = workspace_manager.registry.get_workspace("MLProject")
    assert ml_entry is not None
    assert ml_entry.permission == WorkspacePermission.READ

    target_file = Path(ml_entry.path) / "new_data.py"

    allowed, msg, entry = workspace_manager.validate_file_operation(target_file, WorkspacePermission.WRITE)
    assert allowed is False
    assert "READ-ONLY permission" in msg or "ACCESS DENIED" in msg


# ---------------------------------------------------------------------------
# 2. Path Traversal Prevention (Acceptance Test 6)
# ---------------------------------------------------------------------------


def test_path_traversal_escapes_workspace_blocked(workspace_manager):
    codev_entry = workspace_manager.registry.get_workspace("CodeV")
    assert codev_entry is not None

    # Attempt path traversal escape: CodeV/../../secret
    traversal_path = codev_entry.resolved_path / ".." / ".." / "secret.txt"

    with pytest.raises(PolicyViolationError, match="Path traversal detected"):
        PathGuard.validate_workspace_path(traversal_path, codev_entry.resolved_path)


# ---------------------------------------------------------------------------
# 3. Multiple Workspaces & Directory Router (Acceptance Test 4 & 5)
# ---------------------------------------------------------------------------


def test_directory_router_exact_match(workspace_manager):
    match = workspace_manager.route_task_to_workspace("Fix the authentication bug in CodeV")
    assert match.is_ambiguous is False
    assert match.matched_workspace is not None
    assert match.matched_workspace.name == "CodeV"


def test_directory_router_ambiguous_request(workspace_manager):
    match = workspace_manager.route_task_to_workspace("Fix my Python project")
    assert match.is_ambiguous is True
    assert match.matched_workspace is None
    assert len(match.all_matches) >= 2


# ---------------------------------------------------------------------------
# 4. Project Creation & Templates (Acceptance Test 3)
# ---------------------------------------------------------------------------


def test_create_fastapi_project(workspace_manager):
    pm = ProjectManager(workspace_manager=workspace_manager)
    plan = pm.create_project_plan("Create a FastAPI project called TaskManager", "CodeV")

    assert plan.project_name == "TaskManager"
    assert plan.template_name == "fastapi"
    assert "app/main.py" in plan.planned_files

    success, created_files, report = pm.execute_project_creation(plan)
    assert success is True
    assert len(created_files) > 0
    assert report.is_valid is True
    assert (plan.project_dir / "app" / "main.py").exists()


# ---------------------------------------------------------------------------
# 5. Existing File Protection (Acceptance Test 7)
# ---------------------------------------------------------------------------


def test_existing_file_protection_keep_strategy(workspace_manager):
    agent = ProjectCreationAgent()
    ws_path = workspace_manager.registry.get_workspace("CodeV").path
    plan = agent.parse_requirements_to_plan("Create Python app TaskManager", ws_path)

    # Pre-create main.py with custom content
    main_p = plan.project_dir / "app" / "main.py"
    main_p.parent.mkdir(parents=True, exist_ok=True)
    main_p.write_text("# CUSTOM USER CODE", encoding="utf-8")

    # Check collision
    collisions = agent.check_existing_collisions(plan)
    assert "app/main.py" in collisions

    # Execute with 'keep' strategy -> does NOT overwrite
    agent.execute_plan(plan, overwrite_strategy="keep")
    assert main_p.read_text(encoding="utf-8") == "# CUSTOM USER CODE"


def test_existing_file_protection_backup_strategy(workspace_manager):
    agent = ProjectCreationAgent()
    ws_path = workspace_manager.registry.get_workspace("CodeV").path
    plan = agent.parse_requirements_to_plan("Create Python app TaskManagerBak", ws_path)

    main_p = plan.project_dir / "app" / "main.py"
    main_p.parent.mkdir(parents=True, exist_ok=True)
    main_p.write_text("# ORIGINAL CONTENT", encoding="utf-8")

    agent.execute_plan(plan, overwrite_strategy="backup")

    # Original file is overwritten, backup file created
    assert "# ORIGINAL CONTENT" not in main_p.read_text(encoding="utf-8")
    bak_files = list(main_p.parent.glob("main.py.bak.*"))
    assert len(bak_files) >= 1
    assert bak_files[0].read_text(encoding="utf-8") == "# ORIGINAL CONTENT"


# ---------------------------------------------------------------------------
# 6. Project Inspection
# ---------------------------------------------------------------------------


def test_project_inspection(workspace_manager):
    pm = ProjectManager(workspace_manager=workspace_manager)
    codev_path = workspace_manager.registry.get_workspace("CodeV").path

    profile = pm.inspect_project(codev_path)
    assert profile["name"] == "CodeV"
    assert "Python" in profile["languages"] or "General" in profile["languages"]
    assert profile["workspace_path"] == codev_path


# ---------------------------------------------------------------------------
# 7. V9 Orchestrator Integration Test
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_v9_orchestrator_pipeline(workspace_manager):
    mm = MagicMock(spec=ModelManager)
    mm.chat = MagicMock(side_effect=lambda messages, **kwargs: "Mocked V9 LLM Output")

    orchestrator = V9Orchestrator(
        model_manager=mm,
        workspace_manager=workspace_manager,
    )

    response = await orchestrator.execute_task("Fix the authentication bug in CodeV")

    assert isinstance(response, V9AgentResponse)
    assert response.active_workspace is not None
    assert response.active_workspace.name == "CodeV"
    assert response.workspace_match is not None
    assert response.total_execution_time_sec >= 0.0
