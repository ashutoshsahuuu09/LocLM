"""Project Agent for LocLM Multi-Agent Architecture (V5).

Specialized agent responsible for designing, scaffolding, and writing full project
folder and file structures with complete working implementations.
"""

from __future__ import annotations

import logging
from typing import Any

from loclm.agents.base import AgentContext, AgentResponse, AgentRole, BaseAgent
from loclm.agents.runner import AgentLoop
from loclm.models.base import TaskType
from loclm.models.manager import ModelManager
from loclm.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

PROJECT_AGENT_PROMPT = """You are the LocLM Project Architect and Developer Agent.
Your job is to design, scaffold, and implement complete software projects requested by the user.

WHEN CREATING/PREPARING/DESIGNING A PROJECT:
1. Define the complete folder and file hierarchy (e.g., config, src, tests, README.md, requirements.txt / package.json).
2. Call the `create_project_scaffold` tool OR call `write_file` to create every single file and folder.
3. Write FULL, production-ready, functional code for every file. DO NOT leave placeholders, TODOs, or empty files.
4. Provide a clear summary of the project architecture and created files to the user upon completion.
"""


class ProjectAgent(BaseAgent):
    """Specialized agent for designing and creating full project structures."""

    name = "ProjectAgent"
    role = AgentRole.PROJECT
    description = "Designs, scaffolds, and implements complete project directory and file structures."

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
            max_steps=10,
        )

    async def run(self, context: AgentContext) -> AgentResponse:
        """Run the project agent to design and build the project structure.

        Args:
            context: AgentContext containing the project creation prompt.

        Returns:
            AgentResponse containing the scaffold summary and created files trace.
        """
        logger.info("ProjectAgent executing project creation task: '%s'", context.task[:60])
        return await self._loop.execute_task(context=context, task_type=TaskType.CODING)
