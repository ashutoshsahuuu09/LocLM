"""Specialized Coding Agent for LocLM (V3).

Handles coding tasks: code synthesis, refactoring, bug fixing,
codebase exploration, and repository modification.
"""

from __future__ import annotations

import logging

from loclm.agents.base import AgentContext, AgentResponse, AgentRole, BaseAgent
from loclm.agents.runner import AgentLoop
from loclm.models.base import TaskType
from loclm.models.manager import ModelManager
from loclm.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


class CodingAgent(BaseAgent):
    """Specialized coding agent with codebase inspection and modification tools."""

    name = "coding_agent"
    role = AgentRole.CODING
    description = "Autonomous agent specialized in writing, fixing, refactoring, and searching code."

    def __init__(
        self,
        model_manager: ModelManager,
        tool_registry: ToolRegistry | None = None,
        max_steps: int = 6,
    ) -> None:
        self._model_manager = model_manager
        self._tool_registry = tool_registry or ToolRegistry()
        self._runner = AgentLoop(
            model_manager=self._model_manager,
            tool_registry=self._tool_registry,
            max_steps=max_steps,
        )

    async def run(self, context: AgentContext) -> AgentResponse:
        """Run coding agent on task context.

        Args:
            context: AgentContext containing task details.

        Returns:
            AgentResponse containing solution and execution trace.
        """
        logger.info("CodingAgent started task: '%s'", context.task[:60])
        return await self._runner.execute_task(context, task_type=TaskType.CODING)
