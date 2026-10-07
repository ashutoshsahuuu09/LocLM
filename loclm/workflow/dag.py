"""Autonomous Workflow DAG Engine — LocLM V8.

Provides Directed Acyclic Graph (DAG) task execution with dependency resolution,
topological concurrency grouping, conditional step branching, and execution checkpointing.
"""

from __future__ import annotations

import asyncio
import enum
import logging
import time
from typing import Any, Callable

from pydantic import BaseModel, Field

from loclm.agents.base import AgentContext, AgentResponse, AgentRole, BaseAgent, ExecutionStep

logger = logging.getLogger(__name__)


class StepStatus(str, enum.Enum):
    """Execution status of a workflow step."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class WorkflowStep(BaseModel):
    """A node in the Workflow DAG representing an executable task step."""

    step_id: str = Field(description="Unique step identifier, e.g. step_1")
    instruction: str = Field(description="Natural language instruction for the step")
    assigned_role: AgentRole = Field(
        default=AgentRole.GENERAL,
        description="Target agent role responsible for this step",
    )
    depends_on: list[str] = Field(
        default_factory=list,
        description="List of prerequisite step_ids that must complete before this step runs",
    )
    condition: str | None = Field(
        default=None,
        description="Optional condition string, e.g. 'prev.success' or 'step_1.output'",
    )
    max_retries: int = Field(default=2, description="Maximum retry attempts on step failure")
    status: StepStatus = Field(default=StepStatus.PENDING)
    output: str = Field(default="", description="Output produced by executing this step")
    error: str | None = Field(default=None, description="Error message if execution failed")
    execution_time_sec: float = Field(default=0.0)


class WorkflowState(BaseModel):
    """Serializable snapshot of a Workflow DAG execution state."""

    workflow_id: str = Field(description="Unique identifier for the workflow execution")
    steps: dict[str, WorkflowStep] = Field(default_factory=dict)
    is_completed: bool = Field(default=False)
    has_errors: bool = Field(default=False)
    total_execution_time_sec: float = Field(default=0.0)


class WorkflowDAG:
    """Directed Acyclic Graph structure for multi-step tasks."""

    def __init__(self, workflow_id: str = "default_workflow") -> None:
        self.workflow_id = workflow_id
        self.steps: dict[str, WorkflowStep] = {}

    def add_step(self, step: WorkflowStep) -> None:
        """Add a step node to the DAG.

        Raises:
            ValueError: If step_id already exists.
        """
        if step.step_id in self.steps:
            raise ValueError(f"Duplicate step_id '{step.step_id}' in DAG '{self.workflow_id}'")
        self.steps[step.step_id] = step

    def validate(self) -> None:
        """Validate dependencies and check for cycles in the graph.

        Raises:
            ValueError: If missing dependency or cycle detected.
        """
        # 1. Validate dependencies exist
        for step_id, step in self.steps.items():
            for dep in step.depends_on:
                if dep not in self.steps:
                    raise ValueError(
                        f"Step '{step_id}' references non-existent dependency '{dep}'"
                    )

        # 2. Cycle detection via DFS depth tracking
        visited: dict[str, int] = {s: 0 for s in self.steps}  # 0=unvisited, 1=visiting, 2=visited

        def dfs(node: str) -> None:
            visited[node] = 1
            for dep in self.steps[node].depends_on:
                if visited[dep] == 1:
                    raise ValueError(f"Cycle detected in Workflow DAG involving step '{node}' and '{dep}'")
                if visited[dep] == 0:
                    dfs(dep)
            visited[node] = 2

        for step_id in self.steps:
            if visited[step_id] == 0:
                dfs(step_id)

    def get_topological_levels(self) -> list[list[str]]:
        """Compute topological execution levels where steps in each level can run concurrently.

        Returns:
            List of step_id lists grouped by level index [Level 0, Level 1, ...].
        """
        self.validate()
        in_degree: dict[str, int] = {s: 0 for s in self.steps}
        graph: dict[str, list[str]] = {s: [] for s in self.steps}

        for step_id, step in self.steps.items():
            for dep in step.depends_on:
                graph[dep].append(step_id)
                in_degree[step_id] += 1

        levels: list[list[str]] = []
        current_level = [s for s, deg in in_degree.items() if deg == 0]

        while current_level:
            levels.append(current_level)
            next_level: list[str] = []
            for u in current_level:
                for v in graph[u]:
                    in_degree[v] -= 1
                    if in_degree[v] == 0:
                        next_level.append(v)
            current_level = next_level

        return levels


class WorkflowEngine:
    """Asynchronous engine for executing Workflow DAGs."""

    def __init__(self, agent_map: dict[AgentRole, BaseAgent]) -> None:
        self.agent_map = agent_map

    async def execute_dag(
        self,
        dag: WorkflowDAG,
        user_context: str = "",
    ) -> WorkflowState:
        """Execute all steps in the DAG level by level, running steps in parallel where permitted.

        Args:
            dag: The WorkflowDAG to execute.
            user_context: Global task context string.

        Returns:
            WorkflowState with updated step outputs and execution metadata.
        """
        start_time = time.time()
        dag.validate()
        levels = dag.get_topological_levels()

        logger.info(
            "Executing Workflow DAG '%s': %d steps across %d topological levels",
            dag.workflow_id,
            len(dag.steps),
            len(levels),
        )

        completed_outputs: dict[str, str] = {}
        failed_steps: set[str] = set()

        for level_idx, level_step_ids in enumerate(levels):
            logger.info("DAG Level %d: executing steps %s", level_idx, level_step_ids)
            tasks = []

            for step_id in level_step_ids:
                step = dag.steps[step_id]

                # Check if any prerequisite failed
                if any(dep in failed_steps for dep in step.depends_on):
                    logger.warning("Step '%s' skipped due to failed dependency", step_id)
                    step.status = StepStatus.SKIPPED
                    step.error = "Prerequisite dependency failed"
                    continue

                # Check conditional execution
                if step.condition and not self._evaluate_condition(step.condition, completed_outputs):
                    logger.info("Step '%s' condition '%s' evaluated False → SKIPPED", step_id, step.condition)
                    step.status = StepStatus.SKIPPED
                    continue

                tasks.append(self._execute_step(step, completed_outputs, user_context))

            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for res in results:
                    if isinstance(res, tuple):
                        s_id, s_out, s_err, s_status = res
                        if s_status == StepStatus.COMPLETED:
                            completed_outputs[s_id] = s_out
                        elif s_status == StepStatus.FAILED:
                            failed_steps.add(s_id)

        duration = round(time.time() - start_time, 3)
        has_errors = len(failed_steps) > 0
        all_done = all(s.status in (StepStatus.COMPLETED, StepStatus.SKIPPED) for s in dag.steps.values())

        return WorkflowState(
            workflow_id=dag.workflow_id,
            steps=dag.steps,
            is_completed=all_done,
            has_errors=has_errors,
            total_execution_time_sec=duration,
        )

    async def _execute_step(
        self,
        step: WorkflowStep,
        prev_outputs: dict[str, str],
        user_context: str,
    ) -> tuple[str, str, str | None, StepStatus]:
        """Execute an individual step with retries."""
        step.status = StepStatus.RUNNING
        start_t = time.time()
        agent = self.agent_map.get(step.assigned_role, self.agent_map.get(AgentRole.GENERAL))

        if not agent:
            step.status = StepStatus.FAILED
            step.error = f"No agent registered for role '{step.assigned_role}'"
            return step.step_id, "", step.error, step.status

        # Build context from previous step outputs
        context_parts = [f"Global Task Context: {user_context}"] if user_context else []
        for dep in step.depends_on:
            if dep in prev_outputs:
                context_parts.append(f"Prerequisite [{dep}] Output:\n{prev_outputs[dep]}")

        context_parts.append(f"Current Step Instruction: {step.instruction}")
        full_prompt = "\n\n".join(context_parts)
        ctx = AgentContext(task=full_prompt)

        attempt = 0
        last_err: Exception | None = None

        while attempt <= step.max_retries:
            attempt += 1
            try:
                response = await agent.run(ctx)
                step.output = response.content
                step.status = StepStatus.COMPLETED
                step.execution_time_sec = round(time.time() - start_t, 3)
                logger.info("Step '%s' completed successfully in %.2fs", step.step_id, step.execution_time_sec)
                return step.step_id, step.output, None, step.status
            except Exception as exc:
                last_err = exc
                logger.warning("Step '%s' attempt %d/%d failed: %s", step.step_id, attempt, step.max_retries + 1, exc)

        step.status = StepStatus.FAILED
        step.error = str(last_err) or "Execution failed after retries"
        step.execution_time_sec = round(time.time() - start_t, 3)
        return step.step_id, "", step.error, step.status

    @staticmethod
    def _evaluate_condition(condition: str, prev_outputs: dict[str, str]) -> bool:
        """Simple expression evaluator for step condition strings.

        Supports:
        - "prev.success" or "always"
        - "contains(step_id, 'keyword')"
        - "step_id.output"
        """
        cond = condition.strip().lower()
        if cond in ("always", "prev.success", "true"):
            return True

        if "contains(" in cond:
            try:
                # e.g., contains(step_1, 'success')
                inner = cond.split("contains(", 1)[1].rsplit(")", 1)[0]
                target_step, keyword = [p.strip().strip("'\"") for p in inner.split(",", 1)]
                output = prev_outputs.get(target_step, "")
                return keyword.lower() in output.lower()
            except Exception:
                return True

        # Check if step output exists and is non-empty
        if cond in prev_outputs:
            return bool(prev_outputs[cond].strip())

        return True
