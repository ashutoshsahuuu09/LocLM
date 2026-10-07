"""V8 Multi-Agent Universal Orchestrator — LocLM V8.

V8 extends V7 with four major enterprise architecture upgrades:

1. **Autonomous Security Policy Guard**: Audits tool calls, filesystem paths,
   and terminal commands against configurable zero-trust security rules,
   blocking dangerous patterns and maintaining an immutable audit log.

2. **Autonomous Workflow DAG Engine**: Converts multi-step plans into Directed
   Acyclic Graphs (DAGs) with step dependency resolution, topological level
   concurrency, conditional branching, and checkpoint state snapshots.

3. **Multi-Agent Swarm Collaboration**: For complex or ambiguous queries,
   swarms multiple specialized roles (Leader, Worker, Critic, Verifier) in multi-round
   deliberations to formulate and verify solution outputs.

4. **Multi-Agent Consensus Engine**: Gathers structured proposals from distinct
   agent roles, evaluates agreement scores, extracts dissenting views, and
   synthesizes a high-confidence consensus plan.

Pipeline (V8):
    User Task
      → Policy Guard Audit
      → Session Memory Record
      → Long-term RAG Memory Context Retrieval
      → Router (Intent Classification)
      → Swarm / Consensus (if mode enabled or ambiguous task)
      → Planner → Workflow DAG Construction
      → Workflow Engine Level Execution (Parallel Topological Levels + Step Retries)
      → EvaluatorAgent Quality Pass
      → Session Memory & Long-term Memory Log
      → V8AgentResponse
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from pydantic import BaseModel, Field

from loclm.agents.base import AgentContext, AgentResponse, AgentRole, BaseAgent, ExecutionStep
from loclm.agents.coding import CodingAgent
from loclm.agents.consensus import ConsensusManager, ConsensusResult
from loclm.agents.evaluator_agent import EvaluationResult, EvaluatorAgent
from loclm.agents.general_agent import GeneralAgent
from loclm.agents.knowledge_agent import KnowledgeAgent
from loclm.agents.planner import PlannerAgent, TaskPlan
from loclm.agents.project_agent import ProjectAgent
from loclm.agents.refactor_agent import RefactorAgent
from loclm.agents.router import RouterAgent
from loclm.agents.swarm import SwarmOrchestrator, SwarmResult
from loclm.agents.terminal_agent import TerminalAgent
from loclm.agents.v5_orchestrator import SelfCorrectionTrace
from loclm.memory.manager import MemoryManager
from loclm.memory.session import SessionMemory
from loclm.models.cascade import ModelCascadeManager
from loclm.models.manager import ModelManager
from loclm.plugins.registry import PluginRegistry
from loclm.security.guard import AuditRecord, PolicyGuard, SecurityPolicy
from loclm.tools.registry import ToolRegistry
from loclm.workflow.dag import StepStatus, WorkflowDAG, WorkflowEngine, WorkflowState, WorkflowStep

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# V8 Response Model
# ---------------------------------------------------------------------------


class V8AgentResponse(BaseModel):
    """Complete V8 response object with DAG workflow, swarm, consensus, and security traces."""

    task: str = Field(description="Original user request")
    final_output: str = Field(description="Final response text")
    routed_role: AgentRole = Field(description="Primary agent role selected")
    plan: TaskPlan | None = Field(default=None, description="Planner task plan")

    # V8 Traces
    workflow_state: WorkflowState | None = Field(default=None, description="DAG execution trace")
    consensus_result: ConsensusResult | None = Field(default=None, description="Multi-agent consensus trace")
    swarm_result: SwarmResult | None = Field(default=None, description="Swarm deliberation trace")
    audit_trail: list[AuditRecord] = Field(default_factory=list, description="Security policy audit trail")

    # V7/V5 Traces
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


# ---------------------------------------------------------------------------
# V8 Orchestrator
# ---------------------------------------------------------------------------


class V8Orchestrator:
    """V8 Universal Multi-Agent Orchestrator with Swarm, Workflow DAG, Consensus, and Security Guard.

    Args:
        model_manager: Active ModelManager instance.
        tool_registry: Optional ToolRegistry instance.
        memory_manager: Optional MemoryManager instance.
        session_memory: Optional SessionMemory instance.
        policy_guard: Optional PolicyGuard instance.
        plugin_dir: Directory to scan for runtime plugins.
        max_correction_retries: Max self-correction attempts per step.
        min_quality_threshold: EvaluatorAgent threshold for output quality.
        enable_swarm: Enable multi-agent swarm deliberation on complex tasks.
        enable_consensus: Enable consensus evaluation across agent roles.
        enable_dag_workflow: Execute steps as a Workflow DAG.
    """

    def __init__(
        self,
        model_manager: ModelManager,
        tool_registry: ToolRegistry | None = None,
        memory_manager: MemoryManager | None = None,
        session_memory: SessionMemory | None = None,
        policy_guard: PolicyGuard | None = None,
        plugin_dir: str | None = None,
        max_correction_retries: int = 2,
        min_quality_threshold: float = 0.55,
        enable_swarm: bool = False,
        enable_consensus: bool = False,
        enable_dag_workflow: bool = True,
    ) -> None:
        self.model_manager = model_manager
        self.tool_registry = tool_registry or ToolRegistry()
        self.memory_manager = memory_manager or MemoryManager()
        self.session_memory = session_memory or SessionMemory()
        self.policy_guard = policy_guard or PolicyGuard()
        self.cascade_manager = ModelCascadeManager(self.model_manager)
        self.max_correction_retries = max_correction_retries
        self.min_quality_threshold = min_quality_threshold
        self.enable_swarm = enable_swarm
        self.enable_consensus = enable_consensus
        self.enable_dag_workflow = enable_dag_workflow

        # Core agents
        self.router = RouterAgent(self.model_manager)
        self.planner = PlannerAgent(self.model_manager)
        self.evaluator = EvaluatorAgent(self.model_manager, min_quality_threshold)
        self.consensus_manager = ConsensusManager(self.model_manager)

        # Agent role mapping
        self._agents: dict[AgentRole, BaseAgent] = {
            AgentRole.CODING: CodingAgent(self.model_manager, self.tool_registry),
            AgentRole.PROJECT: ProjectAgent(self.model_manager, self.tool_registry),
            AgentRole.REFACTORING: RefactorAgent(self.model_manager, self.tool_registry),
            AgentRole.TERMINAL: TerminalAgent(self.model_manager, self.tool_registry),
            AgentRole.KNOWLEDGE: KnowledgeAgent(self.model_manager, self.tool_registry),
            AgentRole.GENERAL: GeneralAgent(self.model_manager),
            AgentRole.REASONING: CodingAgent(self.model_manager, self.tool_registry),
        }

        self.swarm_orchestrator = SwarmOrchestrator(self.model_manager, self._agents)

        # Plugin registration
        self._plugin_registry = PluginRegistry(plugin_dir)
        plugin_stats = self._plugin_registry.load_and_register(
            tool_registry=self.tool_registry,
            model_manager=self.model_manager,
            agent_map=self._agents,
        )
        self._plugins_loaded = plugin_stats["plugins"]

        logger.info(
            "V8Orchestrator initialized | session=%s | plugins=%d | security_guard=ACTIVE",
            self.session_memory.session_id[:8],
            self._plugins_loaded,
        )

    async def execute_task(self, user_request: str) -> V8AgentResponse:
        """Execute task through full V8 pipeline.

        Args:
            user_request: The user input prompt.

        Returns:
            V8AgentResponse containing outputs and complete execution traces.
        """
        start_time = time.time()
        logger.info("V8Orchestrator executing task: '%s'", user_request[:80])

        # ── Step 1: Security Audit & Session Memory Record ──────────────────
        self.policy_guard.audit_tool_call("task_execution", {"prompt": user_request[:100]})
        self.session_memory.record_user_message(user_request)

        # ── Step 2: Build enriched prompt ───────────────────────────────────
        session_ctx = self.session_memory.get_session_summary_prefix()
        long_term_ctx = await self.memory_manager.get_relevant_context(user_request, limit=3)
        enriched_prompt = self._build_enriched_prompt(user_request, session_ctx, long_term_ctx)

        # ── Step 3: Route Intent ─────────────────────────────────────────────
        role = await self.router.route(enriched_prompt)
        logger.info("V8 Router selected role: %s", role.value)

        # ── Step 4: Multi-Agent Consensus / Swarm (if enabled) ─────────────
        consensus_res: ConsensusResult | None = None
        swarm_res: SwarmResult | None = None

        if self.enable_consensus:
            consensus_res = await self.consensus_manager.run_consensus(user_request)
            logger.info(
                "V8 Consensus: status=%s | score=%.2f",
                consensus_res.consensus_status.value,
                consensus_res.agreement_score,
            )

        if self.enable_swarm:
            swarm_res = await self.swarm_orchestrator.execute_swarm(user_request, user_context=enriched_prompt)
            logger.info("V8 Swarm completed %d round(s)", swarm_res.rounds_executed)

        # ── Step 5: Plan & Construct Workflow DAG ───────────────────────────
        plan = await self.planner.create_plan(user_request)
        workflow_state: WorkflowState | None = None
        final_output = ""
        execution_steps: list[ExecutionStep] = []

        if self.enable_dag_workflow and plan.steps:
            dag = self._build_dag_from_plan(plan)
            engine = WorkflowEngine(self._agents)
            workflow_state = await engine.execute_dag(dag, user_context=enriched_prompt)

            # Combine outputs from DAG steps
            dag_outputs = [
                s.output for s in workflow_state.steps.values()
                if s.status == StepStatus.COMPLETED and s.output
            ]
            final_output = "\n\n".join(dag_outputs) if dag_outputs else "Workflow execution completed."
        else:
            # Fall back to direct agent run
            agent = self._agents.get(role, self._agents[AgentRole.GENERAL])
            ctx = AgentContext(task=enriched_prompt)
            resp = await agent.run(ctx)
            final_output = resp.content
            execution_steps = list(resp.execution_steps)

        if swarm_res and swarm_res.final_output:
            final_output = f"{swarm_res.final_output}\n\n" + final_output

        # ── Step 6: EvaluatorAgent Quality Pass ─────────────────────────────
        evaluation = await self.evaluator.evaluate(user_request, final_output)
        retry_triggered = False

        if evaluation.needs_retry:
            logger.info("V8 Quality score %.2f < threshold → triggering retry", evaluation.quality_score)
            retry_triggered = True
            retry_prompt = (
                f"{enriched_prompt}\n\n"
                f"[QUALITY REFINEMENT]: {evaluation.improvement_hint}\n"
                f"Please refine and produce an updated answer."
            )
            agent = self._agents.get(role, self._agents[AgentRole.GENERAL])
            retry_ctx = AgentContext(task=retry_prompt)
            try:
                retry_res = await agent.run(retry_ctx)
                final_output = retry_res.content
            except Exception as exc:
                logger.warning("V8 retry failed: %s", exc)

        # ── Step 7: Persist & Log Memory ─────────────────────────────────────
        self.session_memory.record_assistant_message(
            content=final_output[:500],
            metadata={"role": role.value, "v8": True},
        )
        duration = round(time.time() - start_time, 3)
        await self.memory_manager.record_task_run(
            task=user_request,
            result=final_output[:300],
            metadata={"role": role.value, "duration_sec": duration},
        )

        return V8AgentResponse(
            task=user_request,
            final_output=final_output,
            routed_role=role,
            plan=plan,
            workflow_state=workflow_state,
            consensus_result=consensus_res,
            swarm_result=swarm_res,
            audit_trail=self.policy_guard.get_audit_trail(),
            memory_context_used=long_term_ctx,
            session_id=self.session_memory.session_id,
            execution_steps=execution_steps,
            evaluation=evaluation,
            retry_triggered=retry_triggered,
            total_execution_time_sec=duration,
            is_complete=True,
            plugins_loaded=self._plugins_loaded,
        )

    def _build_dag_from_plan(self, plan: TaskPlan) -> WorkflowDAG:
        """Convert a TaskPlan into an executable WorkflowDAG."""
        dag = WorkflowDAG(workflow_id=f"dag_{abs(hash(plan.goal))}")
        prev_id: str | None = None

        for idx, step in enumerate(plan.steps, start=1):
            s_id = f"step_{idx}"
            deps = [prev_id] if prev_id else []
            w_step = WorkflowStep(
                step_id=s_id,
                instruction=step.instruction,
                assigned_role=step.assigned_role,
                depends_on=deps,
            )
            dag.add_step(w_step)
            prev_id = s_id

        return dag

    @staticmethod
    def _build_enriched_prompt(user_request: str, session_ctx: str, long_term_ctx: str) -> str:
        parts = [user_request]
        if session_ctx:
            parts.append(session_ctx)
        if long_term_ctx:
            parts.append(long_term_ctx)
        return "\n\n".join(parts)
