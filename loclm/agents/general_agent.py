"""General Conversational Agent for LocLM (V4).

Handles general conversation, explanation tasks, and direct Q&A.
"""

from __future__ import annotations

import logging

from loclm.agents.base import AgentContext, AgentResponse, AgentRole, BaseAgent
from loclm.models.base import ChatMessage, TaskType
from loclm.models.manager import ModelManager

logger = logging.getLogger(__name__)


class GeneralAgent(BaseAgent):
    """Fast general conversational agent."""

    name = "general_agent"
    role = AgentRole.GENERAL
    description = "Conversational agent for general knowledge, questions, and direct answers."

    def __init__(self, model_manager: ModelManager) -> None:
        self._model_manager = model_manager

    async def run(self, context: AgentContext) -> AgentResponse:
        logger.info("GeneralAgent processing query: '%s'", context.task[:60])
        messages = [
            ChatMessage(role="user", content=context.task)
        ]
        response_text = await self._model_manager.chat(
            messages=messages,
            task_type=TaskType.GENERAL,
            temperature=0.7,
            max_tokens=512,
        )
        return AgentResponse(content=response_text, is_complete=True)
