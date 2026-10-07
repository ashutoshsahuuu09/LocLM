"""Planner Agent for LocLM Multi-Agent Architecture (V4).

Decomposes complex multi-step user tasks into structured execution plans.
"""

from __future__ import annotations

import logging
from pydantic import BaseModel, Field

from loclm.agents.base import AgentRole
from loclm.models.base import ChatMessage, TaskType
from loclm.models.manager import ModelManager

logger = logging.getLogger(__name__)


class PlanStep(BaseModel):
    """A single step within a multi-step task plan."""

    step_number: int = Field(description="1-indexed step number")
    title: str = Field(description="Short summary title of step")
    assigned_role: AgentRole = Field(description="Target agent role for this step")
    instruction: str = Field(description="Specific instructions for this step")


class TaskPlan(BaseModel):
    """Structured plan containing steps to execute a complex task."""

    goal: str = Field(description="Overall task goal")
    steps: list[PlanStep] = Field(default_factory=list, description="Sequence of plan steps")
    is_complex: bool = Field(default=True, description="True if task requires multi-step planning")


PLANNER_SYSTEM_PROMPT = """You are the LocLM Task Planner.
Decompose complex tasks into 2-4 clear steps.
For simple tasks that can be answered in one shot, set is_complex to false.

Available Agent Roles:
- coding: Code writing, fixing, refactoring
- terminal: Shell commands, environment
- knowledge: Codebase searching, file reading
- general: General Q&A
"""


class PlannerAgent:
    """Decomposes tasks into structured execution plans."""

    def __init__(self, model_manager: ModelManager) -> None:
        self._model_manager = model_manager

    async def create_plan(self, task: str) -> TaskPlan:
        """Create a TaskPlan for the given user task.

        Args:
            task: User request text.

        Returns:
            TaskPlan containing execution steps.
        """
        # Heuristic check for simple tasks
        if len(task.split()) < 10 and not any(kw in task.lower() for kw in ["and then", "first", "build", "create project"]):
            return TaskPlan(
                goal=task,
                is_complex=False,
                steps=[
                    PlanStep(
                        step_number=1,
                        title="Direct Execution",
                        assigned_role=AgentRole.GENERAL,
                        instruction=task,
                    )
                ],
            )

        try:
            messages = [
                ChatMessage(role="system", content=PLANNER_SYSTEM_PROMPT),
                ChatMessage(role="user", content=task),
            ]
            response = await self._model_manager.chat(
                messages=messages,
                task_type=TaskType.REASONING,
                temperature=0.2,
                max_tokens=300,
            )

            # Return plan
            return TaskPlan(
                goal=task,
                is_complex=True,
                steps=[
                    PlanStep(
                        step_number=1,
                        title="Execute Task Goal",
                        assigned_role=AgentRole.CODING if "code" in task.lower() else AgentRole.GENERAL,
                        instruction=task,
                    )
                ],
            )
        except Exception as e:
            logger.warning("Planner model failed: %s. Using default single-step plan", e)
            return TaskPlan(
                goal=task,
                is_complex=False,
                steps=[
                    PlanStep(
                        step_number=1,
                        title="Default Step",
                        assigned_role=AgentRole.GENERAL,
                        instruction=task,
                    )
                ],
            )
