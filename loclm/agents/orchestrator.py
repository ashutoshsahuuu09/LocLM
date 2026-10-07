"""Multi-Agent Orchestrator for LocLM (V4).

Coordinates the full multi-agent pipeline:
User Request -> Router -> Planner (if complex) -> Specialized Agent -> Response
"""

from __future__ import annotations

import logging

from loclm.agents.base import AgentContext, AgentResponse, AgentRole, BaseAgent
from loclm.agents.coding import CodingAgent
from loclm.agents.general_agent import GeneralAgent
from loclm.agents.knowledge_agent import KnowledgeAgent
from loclm.agents.planner import PlannerAgent
from loclm.agents.project_agent import ProjectAgent
from loclm.agents.router import RouterAgent
from loclm.agents.terminal_agent import TerminalAgent
from loclm.models.manager import ModelManager
from loclm.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


class AgentOrchestrator:
    """Central multi-agent coordinator and execution pipeline."""

    def __init__(
        self,
        model_manager: ModelManager,
        tool_registry: ToolRegistry | None = None,
    ) -> None:
        self._model_manager = model_manager
        self._tool_registry = tool_registry or ToolRegistry()

        # Components
        self.router = RouterAgent(self._model_manager)
        self.planner = PlannerAgent(self._model_manager)

        # Agent Map
        self._agents: dict[AgentRole, BaseAgent] = {
            AgentRole.CODING: CodingAgent(self._model_manager, self._tool_registry),
            AgentRole.PROJECT: ProjectAgent(self._model_manager, self._tool_registry),
            AgentRole.TERMINAL: TerminalAgent(self._model_manager, self._tool_registry),
            AgentRole.KNOWLEDGE: KnowledgeAgent(self._model_manager, self._tool_registry),
            AgentRole.GENERAL: GeneralAgent(self._model_manager),
            AgentRole.REASONING: CodingAgent(self._model_manager, self._tool_registry),
        }

    async def execute(self, user_request: str) -> AgentResponse:
        """Process user request through the multi-agent pipeline.

        Pipeline:
        1. Classify intent via Router
        2. Plan execution via Planner (if complex)
        3. Dispatch to specialized agent
        4. Return response trace

        Args:
            user_request: The user input text.

        Returns:
            AgentResponse containing final output.
        """
        logger.info("AgentOrchestrator received request: '%s'", user_request[:60])

        # Step 1: Route request
        role = await self.router.route(user_request)
        logger.info("Routed request to agent role: %s", role.value)

        # Step 2: Create context
        context = AgentContext(task=user_request)

        # Step 3: Dispatch to target specialized agent
        agent = self._agents.get(role, self._agents[AgentRole.GENERAL])
        logger.info("Executing task with agent: %s", agent.name)

        return await agent.run(context)
