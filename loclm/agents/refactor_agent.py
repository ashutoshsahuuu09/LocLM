"""Refactor Agent for LocLM Multi-Agent Architecture (V6).

Specialized agent for AST-guided refactoring, safe symbol renaming,
and regression test verification across codebases.
"""

from __future__ import annotations

import logging

from loclm.agents.base import AgentContext, AgentResponse, AgentRole, BaseAgent
from loclm.agents.runner import AgentLoop
from loclm.models.base import TaskType
from loclm.models.manager import ModelManager
from loclm.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


class RefactorAgent(BaseAgent):
    """Specialized agent for AST refactoring and code transformation."""

    name = "RefactorAgent"
    role = AgentRole.REFACTORING
    description = "Handles AST-guided refactoring, safe symbol renaming, and zero-regression code changes."

    def __init__(
        self,
        model_manager: ModelManager,
        tool_registry: ToolRegistry | None = None,
    ) -> None:
        self.model_manager = model_manager
        self.tool_registry = tool_registry or ToolRegistry()
        self._loop = AgentLoop(
            model_manager=self.model_manager,
            tool_registry=self.tool_registry,
            max_steps=8,
        )

    async def run(self, context: AgentContext) -> AgentResponse:
        """Run the refactor agent on the given task context.

        Args:
            context: AgentContext containing refactoring instructions.

        Returns:
            AgentResponse containing output trace and test results.
        """
        logger.info("RefactorAgent executing task: '%s'", context.task[:60])
        return await self._loop.execute_task(context=context, task_type=TaskType.CODING)
