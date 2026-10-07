"""Core orchestrator for LocLM V9.

Handles the complete multi-directory workspace, directory routing, project agent,
swarm collaboration, workflow DAG, and model inference pipeline.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator

from loclm.agents.v9_orchestrator import V9AgentResponse, V9Orchestrator
from loclm.core.state import AppState
from loclm.models.base import ChatMessage, TaskType
from loclm.workspace.manager import WorkspaceManager

logger = logging.getLogger(__name__)


class Orchestrator:
    """Universal LocLM Orchestrator managing workspace context and V9 multi-agent engine."""

    def __init__(self, state: AppState) -> None:
        self._state = state
        self._history: list[ChatMessage] = []
        self._system_message: ChatMessage | None = None
        self.workspace_manager = WorkspaceManager()
        self._v9_orchestrator: V9Orchestrator | None = None

    def setup(self) -> None:
        """Set up the orchestrator with the system prompt and V9 engine."""
        if self._state.config.system_prompt:
            self._system_message = ChatMessage(
                role="system",
                content=self._state.config.system_prompt,
            )

        if self._state.model_manager:
            self._v9_orchestrator = V9Orchestrator(
                model_manager=self._state.model_manager,
                workspace_manager=self.workspace_manager,
            )

    async def chat(self, user_input: str) -> str:
        """Process a user message through V9 orchestrator pipeline.

        Args:
            user_input: The user's message text.

        Returns:
            The final response text.
        """
        if not self._state.model_manager:
            return "Error: Model manager not initialized. Run 'loclm doctor' to check setup."

        if not self._v9_orchestrator:
            self.setup()

        try:
            if self._v9_orchestrator:
                v9_resp: V9AgentResponse = await self._v9_orchestrator.execute_task(user_input)
                response = v9_resp.final_output
            else:
                messages = self._build_messages()
                messages.append(ChatMessage(role="user", content=user_input))
                response = await self._state.model_manager.chat(messages=messages, temperature=self._state.config.temperature)

            self._history.append(ChatMessage(role="user", content=user_input))
            self._history.append(ChatMessage(role="assistant", content=response))
            self._trim_history()

            return response

        except Exception as e:
            logger.error("V9 Orchestrator chat failed: %s", e)
            return f"Error executing request: {e}"

    async def chat_stream(self, user_input: str) -> AsyncIterator[str]:
        """Process a user message and stream response."""
        full_res = await self.chat(user_input)
        yield full_res

    def clear_history(self) -> None:
        """Clear conversation history."""
        self._history.clear()
        logger.info("Conversation history cleared")

    @property
    def history_length(self) -> int:
        """Number of messages in history."""
        return len(self._history)

    def _build_messages(self) -> list[ChatMessage]:
        messages: list[ChatMessage] = []
        if self._system_message:
            messages.append(self._system_message)
        messages.extend(self._history)
        return messages

    def _trim_history(self) -> None:
        max_messages = self._state.config.max_context_messages
        if len(self._history) > max_messages:
            excess = len(self._history) - max_messages
            self._history = self._history[excess:]
