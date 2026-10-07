"""Hardware-aware Model Cascade & Dynamic Offloading for LocLM V5.

Provides multi-model cascading fallback: if execution fails or memory limits
are reached on a primary model, falls back gracefully to secondary model tiers.
"""

from __future__ import annotations

import logging
from typing import Any

from loclm.models.base import ChatMessage, TaskType
from loclm.models.manager import ModelManager, ModelNotAvailableError

logger = logging.getLogger(__name__)


class ModelCascadeManager:
    """Manages model cascading and fallback execution strategy."""

    def __init__(self, model_manager: ModelManager) -> None:
        self.model_manager = model_manager

    async def chat_with_cascade(
        self,
        messages: list[ChatMessage],
        task_type: TaskType = TaskType.GENERAL,
        temperature: float = 0.5,
        max_tokens: int | None = None,
    ) -> str:
        """Execute chat request with automatic fallback cascade.

        Args:
            messages: List of input ChatMessages.
            task_type: Preferred task type for primary model selection.
            temperature: Sampling temperature.
            max_tokens: Maximum response token count.

        Returns:
            Model response text.
        """
        # Primary attempt using task-specific model
        try:
            return await self.model_manager.chat(
                messages=messages,
                task_type=task_type,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as primary_error:
            logger.warning(
                "Primary model inference failed (%s). Cascading to GENERAL fallback model...",
                primary_error,
            )

        # Secondary fallback attempt using GENERAL model
        try:
            return await self.model_manager.chat(
                messages=messages,
                task_type=TaskType.GENERAL,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as fallback_error:
            logger.error("Cascade fallback model inference also failed: %s", fallback_error)
            raise ModelNotAvailableError(
                f"Model cascade failed. Primary error: {primary_error}, Fallback error: {fallback_error}"
            ) from fallback_error
