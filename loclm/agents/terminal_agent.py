"""Specialized Terminal Agent for LocLM (V4).

Handles shell execution, environment inspection, git operations, and system tasks.
"""

from __future__ import annotations

import logging

from loclm.agents.base import AgentContext, AgentResponse, AgentRole, BaseAgent
from loclm.agents.runner import AgentLoop
from loclm.models.base import TaskType
from loclm.models.manager import ModelManager
from loclm.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


class TerminalAgent(BaseAgent):
    """Specialized agent for terminal command execution and Git operations."""

    name = "terminal_agent"
    role = AgentRole.TERMINAL
    description = "Autonomous agent specialized in shell execution, Git, and environment operations."

    def __init__(
        self,
        model_manager: ModelManager,
        tool_registry: ToolRegistry | None = None,
        max_steps: int = 5,
    ) -> None:
        self._model_manager = model_manager
        self._tool_registry = tool_registry or ToolRegistry()
        self._runner = AgentLoop(
            model_manager=self._model_manager,
            tool_registry=self._tool_registry,
            max_steps=max_steps,
        )

    async def run(self, context: AgentContext) -> AgentResponse:
        logger.info("TerminalAgent started task: '%s'", context.task[:60])
        return await self._runner.execute_task(context, task_type=TaskType.GENERAL)
