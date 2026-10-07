"""V9 Multi-Agent Universal Workspace Orchestrator — LocLM V9.

V9 extends V8 with multi-directory workspace management, directory routing,
workspace permission enforcement (READ, WRITE, FULL), project scaffolding engine (ProjectCreationAgent),
and zero-trust operation auditing.

Pipeline (V9):
    User Request
      → Workspace Router (Directory Match / Ambiguity Check)
      → Workspace Permission & Security Audit (PathGuard / PermissionGuard)
      → Session Memory Record & RAG Long-Term Memory
      → Router (Intent Classification)
      → ProjectCreationAgent (if project creation requested)
      → Swarm / Consensus (if enabled)
      → Workflow DAG Engine Level Execution
      → EvaluatorAgent Quality Pass
      → V9AgentResponse
"""

from __future__ import annotations

import logging
import time
from typing import Any

from pydantic import BaseModel, Field

from loclm.agents.base import AgentContext, AgentResponse, AgentRole, BaseAgent, ExecutionStep
from loclm.agents.consensus import ConsensusResult
from loclm.agents.evaluator_agent import EvaluationResult
from loclm.agents.planner import TaskPlan
from loclm.agents.swarm import SwarmResult
from loclm.agents.v5_orchestrator import SelfCorrectionTrace
from loclm.agents.v8_orchestrator import V8Orchestrator
from loclm.models.manager import ModelManager
from loclm.projects.creator import ProjectPlan
from loclm.projects.manager import ProjectManager
from loclm.security.guard import AuditRecord, PolicyGuard
from loclm.security.operation_guard import OperationGuard
from loclm.workspace.manager import WorkspaceManager
from loclm.workspace.permissions import WorkspacePermission
from loclm.workspace.registry import WorkspaceEntry
from loclm.workspace.router import WorkspaceMatch
from loclm.workflow.dag import WorkflowState

logger = logging.getLogger(__name__)


class V9AgentResponse(BaseModel):
    """Complete V9 response object with multi-directory workspace, routing, and project traces."""

    task: str = Field(description="Original user request")
    final_output: str = Field(description="Final response text")
    routed_role: AgentRole = Field(description="Primary agent role selected")

    # V9 Traces
    active_workspace: WorkspaceEntry | None = Field(default=None, description="Active workspace context")
    workspace_match: WorkspaceMatch | None = Field(default=None, description="Directory router match result")
    project_plan: ProjectPlan | None = Field(default=None, description="Project creation plan if applicable")

    # V8/V7/V5 Traces
    plan: TaskPlan | None = Field(default=None)
    workflow_state: WorkflowState | None = Field(default=None)
    consensus_result: ConsensusResult | None = Field(default=None)
    swarm_result: SwarmResult | None = Field(default=None)
    audit_trail: list[AuditRecord] = Field(default_factory=list)

    # Memory
    memory_context_used: str = Field(default="")
    session_id: str = Field(default="")
    execution_steps: list[ExecutionStep] = Field(default_factory=list)
    self_corrections: list[SelfCorrectionTrace] = Field(default_factory=list)
    evaluation: EvaluationResult | None = Field(default=None)
    retry_triggered: bool = Field(default=False)

    # Meta
    total_execution_time_sec: float = Field(default=0.0)
    is_complete: bool = Field(default=True)
    plugins_loaded: int = Field(default=0)


class V9Orchestrator:
    """V9 Universal Orchestrator with Multi-Directory Workspaces, Directory Router, and Project Agent."""

    def __init__(
        self,
        model_manager: ModelManager,
        workspace_manager: WorkspaceManager | None = None,
        project_manager: ProjectManager | None = None,
        policy_guard: PolicyGuard | None = None,
        plugin_dir: str | None = None,
        enable_swarm: bool = False,
        enable_consensus: bool = False,
        enable_dag_workflow: bool = True,
    ) -> None:
        self.model_manager = model_manager
        self.workspace_manager = workspace_manager or WorkspaceManager()
        self.project_manager = project_manager or ProjectManager(self.workspace_manager, self.model_manager)
        self.policy_guard = policy_guard or PolicyGuard()
        self.operation_guard = OperationGuard(self.policy_guard)

        # Base V8 Orchestrator
        self.v8_orchestrator = V8Orchestrator(
            model_manager=self.model_manager,
            policy_guard=self.policy_guard,
            plugin_dir=plugin_dir,
            enable_swarm=enable_swarm,
            enable_consensus=enable_consensus,
            enable_dag_workflow=enable_dag_workflow,
        )

        logger.info(
            "V9Orchestrator initialized | workspaces=%d | active_ws=%s",
            len(self.workspace_manager.list_workspaces()),
            self.workspace_manager.context.workspace_name,
        )

    async def execute_task(
        self,
        user_request: str,
        target_workspace: str | None = None,
    ) -> V9AgentResponse:
        """Execute user request through full V9 workspace pipeline.

        Args:
            user_request: User request prompt text.
            target_workspace: Optional explicit workspace name or path.

        Returns:
            V9AgentResponse with complete workspace, project, and execution traces.
        """
        start_time = time.time()
        logger.info("V9Orchestrator task: '%s'", user_request[:80])

        # ── Step 1: Directory Router & Workspace Match ──────────────────────
        if target_workspace:
            ws_entry = self.workspace_manager.use_workspace(target_workspace)
            ws_match = WorkspaceMatch(
                matched_workspace=ws_entry,
                all_matches=[ws_entry],
                is_ambiguous=False,
                match_reason=f"Explicitly specified workspace '{ws_entry.name}'",
            )
        else:
            ws_match = self.workspace_manager.route_task_to_workspace(user_request)

        # Handle ambiguous directory requests
        if ws_match.is_ambiguous and not ws_match.matched_workspace:
            ws_list_str = "\n".join(f"{idx+1}. {w.name} ({w.path})" for idx, w in enumerate(ws_match.all_matches))
            ambiguous_output = (
                f"I found multiple matching approved workspaces:\n\n{ws_list_str}\n\n"
                f"Which workspace should I use? Please specify by name."
            )
            return V9AgentResponse(
                task=user_request,
                final_output=ambiguous_output,
                routed_role=AgentRole.GENERAL,
                workspace_match=ws_match,
                active_workspace=None,
                total_execution_time_sec=round(time.time() - start_time, 3),
                is_complete=False,
            )

        active_ws = self.workspace_manager.context.active_workspace

        # ── Step 2: Check for Project Creation Request ──────────────────────
        project_plan: ProjectPlan | None = None
        req_lower = user_request.lower()
        if "create" in req_lower and any(kw in req_lower for kw in ("project", "backend", "app", "fastapi", "react", "vite", "cli", "dashboard")):
            if active_ws:
                try:
                    project_plan = self.project_manager.create_project_plan(user_request, active_ws.name)
                    logger.info("V9 ProjectPlan created: '%s' at '%s'", project_plan.project_name, project_plan.project_dir)
                except Exception as exc:
                    logger.warning("V9 ProjectPlan creation warning: %s", exc)

        # ── Step 3: Execute via V8 Orchestrator Pipeline ────────────────────
        v8_resp = await self.v8_orchestrator.execute_task(user_request)

        final_output = v8_resp.final_output

        # If project creation was planned, append project report
        if project_plan:
            plan_report = project_plan.format_plan_report()
            final_output = f"{plan_report}\n\n" + final_output

        duration = round(time.time() - start_time, 3)

        return V9AgentResponse(
            task=user_request,
            final_output=final_output,
            routed_role=v8_resp.routed_role,
            active_workspace=active_ws,
            workspace_match=ws_match,
            project_plan=project_plan,
            plan=v8_resp.plan,
            workflow_state=v8_resp.workflow_state,
            consensus_result=v8_resp.consensus_result,
            swarm_result=v8_resp.swarm_result,
            audit_trail=self.policy_guard.get_audit_trail(),
            memory_context_used=v8_resp.memory_context_used,
            session_id=v8_resp.session_id,
            execution_steps=v8_resp.execution_steps,
            self_corrections=v8_resp.self_corrections,
            evaluation=v8_resp.evaluation,
            retry_triggered=v8_resp.retry_triggered,
            total_execution_time_sec=duration,
            is_complete=v8_resp.is_complete,
            plugins_loaded=v8_resp.plugins_loaded,
        )
