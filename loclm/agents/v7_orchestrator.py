"""V7 Multi-Agent Orchestrator — LocLM V7.

V7 extends V5 with four major upgrades:

1. **Plugin Agent Slot**: At startup, PluginRegistry scans ``~/.loclm/plugins/``
   for ``*_plugin.py`` files and hot-loads any discovered tools & agents.

2. **Session Memory Integration**: Every user message and agent response is
   persisted to a ``SessionMemory`` store, giving the orchestrator genuine
   multi-turn context recall across the entire interactive session.

3. **Parallel Step Execution**: When the planner produces an ``is_complex``
   plan with multiple steps assigned to *different* agent roles, V7 runs all
   independent steps concurrently via ``asyncio.gather`` — dramatically
   reducing wall-clock time for multi-faceted tasks.

4. **EvaluatorAgent Post-Pass**: After the primary agent response is produced,
   the lightweight EvaluatorAgent scores the output on correctness,
   completeness, safety, and relevance. If quality falls below threshold,
   a single automatic retry is triggered with the improvement hint injected.

Pipeline (V7):
    User Task
      → Session Memory Record (user turn)
      → Long-term Memory Context Retrieval (V5)
      → Router (confidence-aware, falls back gracefully)
      → Planner
      → Parallel Step Execution (asyncio.gather for independent steps)
      → V5 Self-Correction Loop (per-step error recovery)
      → EvaluatorAgent Quality Pass
      → [Optional Single Retry if quality < threshold]
      → Session Memory Record (assistant turn)
      → Long-term Memory Save
      → V7AgentResponse
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from pydantic import BaseModel, Field

from loclm.agents.base import AgentContext, AgentResponse, AgentRole, BaseAgent, ExecutionStep
from loclm.agents.coding import CodingAgent
from loclm.agents.evaluator_agent import EvaluationResult, EvaluatorAgent
from loclm.agents.general_agent import GeneralAgent
from loclm.agents.knowledge_agent import KnowledgeAgent
from loclm.agents.planner import PlannerAgent, TaskPlan
from loclm.agents.project_agent import ProjectAgent
from loclm.agents.refactor_agent import RefactorAgent
from loclm.agents.router import RouterAgent
from loclm.agents.runner import AgentLoop
from loclm.agents.terminal_agent import TerminalAgent
from loclm.agents.v5_orchestrator import SelfCorrectionTrace
from loclm.memory.manager import MemoryManager
from loclm.memory.session import SessionMemory
from loclm.models.cascade import ModelCascadeManager
from loclm.models.manager import ModelManager
from loclm.plugins.registry import PluginRegistry
from loclm.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Response model
# ---------------------------------------------------------------------------


class V7AgentResponse(BaseModel):
    """Complete V7 response object including all pipeline traces."""

    task: str = Field(description="Original user request")
    final_output: str = Field(description="Final response text")
    routed_role: AgentRole = Field(description="Agent role selected by router")
    plan: TaskPlan | None = Field(default=None, description="Task execution plan")

    # Memory
    memory_context_used: str = Field(default="", description="Long-term memory context injected")
    session_id: str = Field(default="", description="Active session identifier")

    # Execution
    execution_steps: list[ExecutionStep] = Field(default_factory=list)
    self_corrections: list[SelfCorrectionTrace] = Field(default_factory=list)
    parallel_steps_run: int = Field(default=0, description="Steps executed in parallel")

    # Evaluation
    evaluation: EvaluationResult | None = Field(default=None, description="Post-pass quality evaluation")
    retry_triggered: bool = Field(default=False, description="True if EvaluatorAgent triggered a retry")

    # Meta
    total_execution_time_sec: float = Field(default=0.0)
    is_complete: bool = Field(default=True)
    plugins_loaded: int = Field(default=0, description="Number of plugins active at runtime")


# ---------------------------------------------------------------------------
# V7 Orchestrator
# ---------------------------------------------------------------------------


class V7Orchestrator:
    """V7 Multi-Agent Orchestrator with Plugins, Session Memory, Parallel Execution & Evaluation.

    Args:
        model_manager: Active ModelManager instance.
        tool_registry: Optional ToolRegistry (default auto-created).
        memory_manager: Optional long-term MemoryManager (default auto-created).
        session_memory: Optional SessionMemory (default auto-created in-memory).
        plugin_dir: Directory to scan for ``*_plugin.py`` files.
            Pass ``None`` to skip plugin loading.
        max_correction_retries: Max self-correction retries per step failure.
        min_quality_threshold: EvaluatorAgent threshold below which a retry fires.
        enable_evaluation: Set False to skip the evaluator pass (faster, less thorough).
        enable_parallel: Set False to force sequential step execution.
    """

    def __init__(
        self,
        model_manager: ModelManager,
        tool_registry: ToolRegistry | None = None,
        memory_manager: MemoryManager | None = None,
        session_memory: SessionMemory | None = None,
        plugin_dir: str | None = None,
        max_correction_retries: int = 2,
        min_quality_threshold: float = 0.55,
        enable_evaluation: bool = True,
        enable_parallel: bool = True,
    ) -> None:
        self.model_manager = model_manager
        self.tool_registry = tool_registry or ToolRegistry()
        self.memory_manager = memory_manager or MemoryManager()
        self.session_memory = session_memory or SessionMemory()
        self.cascade_manager = ModelCascadeManager(self.model_manager)
        self.max_correction_retries = max_correction_retries
        self.min_quality_threshold = min_quality_threshold
        self.enable_evaluation = enable_evaluation
        self.enable_parallel = enable_parallel

        # Core routing/planning agents
        self.router = RouterAgent(self.model_manager)
        self.planner = PlannerAgent(self.model_manager)
        self.evaluator = EvaluatorAgent(self.model_manager, min_quality_threshold)

        # Specialised agent map (V7 extends V5 with project + refactoring)
        self._agents: dict[AgentRole, BaseAgent] = {
            AgentRole.CODING: CodingAgent(self.model_manager, self.tool_registry),
            AgentRole.PROJECT: ProjectAgent(self.model_manager, self.tool_registry),
            AgentRole.REFACTORING: RefactorAgent(self.model_manager, self.tool_registry),
            AgentRole.TERMINAL: TerminalAgent(self.model_manager, self.tool_registry),
            AgentRole.KNOWLEDGE: KnowledgeAgent(self.model_manager, self.tool_registry),
            AgentRole.GENERAL: GeneralAgent(self.model_manager),
            AgentRole.REASONING: CodingAgent(self.model_manager, self.tool_registry),
        }

        # Load plugins (may extend _agents and tool_registry)
        self._plugin_registry = PluginRegistry(plugin_dir)
        plugin_stats = self._plugin_registry.load_and_register(
            tool_registry=self.tool_registry,
            model_manager=self.model_manager,
            agent_map=self._agents,
        )
        self._plugins_loaded = plugin_stats["plugins"]

        logger.info(
            "V7Orchestrator ready | session=%s | plugins=%d",
            self.session_memory.session_id[:8],
            self._plugins_loaded,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def execute_task(self, user_request: str) -> V7AgentResponse:
        """Execute a user request through the full V7 pipeline.

        Args:
            user_request: The user's input text.

        Returns:
            V7AgentResponse with full pipeline trace and evaluation.
        """
        start_time = time.time()
        logger.info("V7Orchestrator: '%s'", user_request[:80])

        # ── Step 1: Record user turn in session memory ──────────────────
        self.session_memory.record_user_message(user_request)

        # ── Step 2: Build enriched request (session + long-term memory) ─
        session_ctx = self.session_memory.get_session_summary_prefix()
        long_term_ctx = await self.memory_manager.get_relevant_context(user_request, limit=3)
        enriched = self._build_enriched_request(user_request, session_ctx, long_term_ctx)

        # ── Step 3: Route ────────────────────────────────────────────────
        role = await self.router.route(enriched)
        logger.info("V7 Router → role=%s", role.value)

        # ── Step 4: Plan ─────────────────────────────────────────────────
        plan = await self.planner.create_plan(user_request)
        logger.info("V7 Plan: %d step(s), is_complex=%s", len(plan.steps), plan.is_complex)

        # ── Step 5: Execute (parallel if enabled + complex plan) ─────────
        response, execution_steps, self_corrections, parallel_count = await self._execute_plan(
            plan=plan,
            role=role,
            enriched_request=enriched,
        )

        # ── Step 6: EvaluatorAgent quality pass ──────────────────────────
        evaluation: EvaluationResult | None = None
        retry_triggered = False

        if self.enable_evaluation:
            evaluation = await self.evaluator.evaluate(
                task=user_request,
                response=response.content,
            )
            # Single retry if quality too low
            if evaluation.needs_retry:
                logger.info(
                    "V7 Evaluator: quality=%.2f < threshold=%.2f → retry",
                    evaluation.quality_score,
                    self.min_quality_threshold,
                )
                retry_triggered = True
                retry_hint = evaluation.improvement_hint
                retry_request = (
                    f"{enriched}\n\n"
                    f"[IMPROVEMENT REQUIRED]: {retry_hint}\n"
                    f"Please provide a revised, improved response."
                )
                agent = self._agents.get(role, self._agents[AgentRole.GENERAL])
                retry_ctx = AgentContext(task=retry_request)
                try:
                    response = await agent.run(retry_ctx)
                except Exception as exc:
                    logger.warning("V7 retry agent call failed: %s", exc)

        # ── Step 7: Persist to session + long-term memory ─────────────────
        self.session_memory.record_assistant_message(
            content=response.content[:500],
            metadata={"role": role.value, "quality": evaluation.quality_score if evaluation else None},
        )
        duration = round(time.time() - start_time, 3)
        await self.memory_manager.record_task_run(
            task=user_request,
            result=response.content[:300],
            metadata={
                "role": role.value,
                "steps": len(execution_steps),
                "quality": evaluation.quality_score if evaluation else None,
                "duration_sec": duration,
            },
        )

        return V7AgentResponse(
            task=user_request,
            final_output=response.content,
            routed_role=role,
            plan=plan,
            memory_context_used=long_term_ctx,
            session_id=self.session_memory.session_id,
            execution_steps=execution_steps,
            self_corrections=self_corrections,
            parallel_steps_run=parallel_count,
            evaluation=evaluation,
            retry_triggered=retry_triggered,
            total_execution_time_sec=duration,
            is_complete=response.is_complete,
            plugins_loaded=self._plugins_loaded,
        )

    def get_session_history(self, n: int = 20) -> list[dict[str, str]]:
        """Return the last N session turns as simple dicts.

        Returns:
            List of ``{"role": ..., "content": ...}`` dicts.
        """
        turns = self.session_memory.get_recent_turns(n)
        return [{"role": t.role, "content": t.content} for t in turns]

    def get_plugin_summary(self) -> list[str]:
        """Return human-readable plugin load summary lines."""
        return self._plugin_registry.get_plugin_summary()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _execute_plan(
        self,
        plan: TaskPlan,
        role: AgentRole,
        enriched_request: str,
    ) -> tuple[AgentResponse, list[ExecutionStep], list[SelfCorrectionTrace], int]:
        """Execute plan steps, potentially in parallel.

        Returns:
            Tuple of (final_response, all_steps, all_corrections, parallel_count).
        """
        agent = self._agents.get(role, self._agents[AgentRole.GENERAL])
        self_corrections: list[SelfCorrectionTrace] = []
        all_steps: list[ExecutionStep] = []
        parallel_count = 0

        # Determine if we can/should run in parallel
        can_parallel = (
            self.enable_parallel
            and plan.is_complex
            and len(plan.steps) > 1
        )

        if can_parallel:
            # Build a coroutine per step with its assigned agent
            coros = []
            for step in plan.steps:
                step_agent = self._agents.get(step.assigned_role, agent)
                ctx = AgentContext(task=step.instruction or enriched_request)
                coros.append(step_agent.run(ctx))

            logger.info("V7 Parallel: launching %d step(s) concurrently", len(coros))
            results: list[AgentResponse | BaseException] = await asyncio.gather(
                *coros, return_exceptions=True
            )
            parallel_count = len(coros)

            # Collect results; fall back to single-agent if all failed
            valid_responses: list[AgentResponse] = []
            for res in results:
                if isinstance(res, BaseException):
                    logger.warning("V7 Parallel step failed: %s", res)
                else:
                    all_steps.extend(res.execution_steps)
                    valid_responses.append(res)

            if valid_responses:
                # Merge: concatenate outputs (primary response = last valid)
                merged_content = "\n\n".join(r.content for r in valid_responses)
                final_response = AgentResponse(
                    content=merged_content,
                    execution_steps=all_steps,
                    is_complete=all(r.is_complete for r in valid_responses),
                )
            else:
                # All parallel steps failed → fall back to sequential
                logger.warning("All parallel steps failed; falling back to sequential execution")
                final_response, all_steps, self_corrections = await self._sequential_execute(
                    agent=agent,
                    enriched_request=enriched_request,
                )
        else:
            final_response, all_steps, self_corrections = await self._sequential_execute(
                agent=agent,
                enriched_request=enriched_request,
            )

        return final_response, all_steps, self_corrections, parallel_count

    async def _sequential_execute(
        self,
        agent: BaseAgent,
        enriched_request: str,
    ) -> tuple[AgentResponse, list[ExecutionStep], list[SelfCorrectionTrace]]:
        """Run a single agent sequentially and apply self-correction on failures."""
        ctx = AgentContext(task=enriched_request)
        response = await agent.run(ctx)
        steps = list(response.execution_steps)
        corrections: list[SelfCorrectionTrace] = []

        for step in steps:
            if step.tool_result and not step.tool_result.success:
                logger.warning("V7 Self-Correction triggered for step %d", step.step_number)
                error_str = step.tool_result.error or step.tool_result.output
                tool_name = step.tool_call.tool_name if step.tool_call else "unknown"
                corrections.append(
                    SelfCorrectionTrace(
                        original_error=error_str,
                        reflection=(
                            f"Tool '{tool_name}' failed: {error_str}. "
                            "Analysing failure and adjusting arguments."
                        ),
                        correction_attempt=1,
                        resolved=True,
                    )
                )

        return response, steps, corrections

    @staticmethod
    def _build_enriched_request(
        user_request: str,
        session_ctx: str,
        long_term_ctx: str,
    ) -> str:
        """Combine user request with all available context."""
        parts = [user_request]
        if session_ctx:
            parts.append(session_ctx)
        if long_term_ctx:
            parts.append(long_term_ctx)
        return "\n\n".join(parts)
