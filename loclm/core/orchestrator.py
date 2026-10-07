"""Core orchestrator for LocLM V1.

Handles the basic chat loop:
User message -> system prompt + history -> model -> response

In later versions, this will evolve into the full
Router -> Planner -> Agent -> Tools -> Memory -> Verification pipeline.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator

from loclm.core.state import AppState
from loclm.models.base import ChatMessage, TaskType

logger = logging.getLogger(__name__)


class Orchestrator:
    """V1 Orchestrator -- manages the chat conversation loop.

    Maintains conversation history and routes messages through
    the model manager for inference.
    """

    def __init__(self, state: AppState) -> None:
        self._state = state
        self._history: list[ChatMessage] = []
        self._system_message: ChatMessage | None = None

    def setup(self) -> None:
        """Set up the orchestrator with the system prompt."""
        if self._state.config.system_prompt:
            self._system_message = ChatMessage(
                role="system",
                content=self._state.config.system_prompt,
            )

    async def chat(self, user_input: str) -> str:
        """Process a user message and return the model's response.

        Args:
            user_input: The user's message text.

        Returns:
            The model's response text.
        """
        if not self._state.model_manager:
            return "Error: Model manager not initialized. Run 'loclm doctor' to check setup."

        # Add user message to history
        self._history.append(ChatMessage(role="user", content=user_input))

        # Build messages: system + trimmed history
        messages = self._build_messages()

        # Get response from model
        try:
            response = await self._state.model_manager.chat(
                messages=messages,
                task_type=TaskType.GENERAL,
                temperature=self._state.config.temperature,
                max_tokens=self._state.config.max_tokens,
            )

            # Add assistant response to history
            self._history.append(ChatMessage(role="assistant", content=response))

            # Trim history if too long
            self._trim_history()

            return response

        except Exception as e:
            logger.error("Chat failed: %s", e)
            error_msg = f"Error: {e}"
            return error_msg

    async def chat_stream(self, user_input: str) -> AsyncIterator[str]:
        """Process a user message and stream the model's response.

        Args:
            user_input: The user's message text.

        Yields:
            Response tokens/chunks as they arrive.
        """
        if not self._state.model_manager:
            yield "Error: Model manager not initialized. Run 'loclm doctor' to check setup."
            return

        # Add user message to history
        self._history.append(ChatMessage(role="user", content=user_input))

        # Build messages
        messages = self._build_messages()

        # Stream response
        full_response: list[str] = []
        try:
            async for token in self._state.model_manager.chat_stream(
                messages=messages,
                task_type=TaskType.GENERAL,
                temperature=self._state.config.temperature,
                max_tokens=self._state.config.max_tokens,
            ):
                full_response.append(token)
                yield token

            # Save complete response to history
            complete = "".join(full_response)
            self._history.append(ChatMessage(role="assistant", content=complete))

            # Trim history
            self._trim_history()

        except Exception as e:
            logger.error("Streaming chat failed: %s", e)
            yield f"\n\nError: {e}"

    def clear_history(self) -> None:
        """Clear conversation history."""
        self._history.clear()
        logger.info("Conversation history cleared")

    @property
    def history_length(self) -> int:
        """Number of messages in history."""
        return len(self._history)

    def _build_messages(self) -> list[ChatMessage]:
        """Build the full message list: system + history."""
        messages: list[ChatMessage] = []

        if self._system_message:
            messages.append(self._system_message)

        messages.extend(self._history)
        return messages

    def _trim_history(self) -> None:
        """Trim conversation history to stay within limits."""
        max_messages = self._state.config.max_context_messages
        if len(self._history) > max_messages:
            # Keep the most recent messages
            excess = len(self._history) - max_messages
            self._history = self._history[excess:]
            logger.debug("Trimmed %d messages from history", excess)
