"""Specialized Knowledge Agent for LocLM (V4).

Handles codebase search, document reading, and repository structure analysis.
"""

from __future__ import annotations

import logging

from loclm.agents.base import AgentContext, AgentResponse, AgentRole, BaseAgent
from loclm.agents.runner import AgentLoop
from loclm.models.base import TaskType
from loclm.models.manager import ModelManager
from loclm.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


class KnowledgeAgent(BaseAgent):
    """Specialized agent for repository inspection, file search, and documentation."""

    name = "knowledge_agent"
    role = AgentRole.KNOWLEDGE
    description = "Autonomous agent specialized in searching codebases, reading files, and Q&A."

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
        logger.info("KnowledgeAgent started task: '%s'", context.task[:60])
        return await self._runner.execute_task(context, task_type=TaskType.GENERAL)
