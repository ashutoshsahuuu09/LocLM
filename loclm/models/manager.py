"""Model manager module.

Central controller for model lifecycle:
- Discovery of installed models
- Selection of optimal model per task
- Loading and unloading models
- Resource monitoring to prevent exhaustion
"""

from __future__ import annotations

import logging
from typing import Any

from loclm.hardware.profiler import HardwareProfile
from loclm.models.base import ChatMessage, ModelInfo, ModelRuntime, TaskType
from loclm.models.selector import ModelSelector

logger = logging.getLogger(__name__)


class ModelManager:
    """Manages model lifecycle and selection for LocLM.

    Responsibilities:
    - Discover installed models from the runtime
    - Recommend and select models based on hardware + task
    - Track the currently active model
    - Load/unload models to manage resources
    - Prevent resource exhaustion (never load a model too large for hardware)
    """

    def __init__(
        self,
        runtime: ModelRuntime,
        profile: HardwareProfile,
        selector: ModelSelector | None = None,
    ) -> None:
        self._runtime = runtime
        self._profile = profile
        self._selector = selector or ModelSelector()

        # State
        self._available_models: list[ModelInfo] = []
        self._active_model: str | None = None
        self._model_cache: dict[str, str] = {}  # task_type -> selected model name

    @property
    def runtime(self) -> ModelRuntime:
        """The underlying model runtime."""
        return self._runtime

    @property
    def profile(self) -> HardwareProfile:
        """Hardware profile used for model selection."""
        return self._profile

    @property
    def active_model(self) -> str | None:
        """Currently active model name."""
        return self._active_model

    @property
    def available_models(self) -> list[ModelInfo]:
        """List of available (installed) models."""
        return self._available_models

    async def initialize(self) -> bool:
        """Initialize the model manager: check runtime and discover models.

        Returns:
            True if initialization succeeded, False otherwise.
        """
        # Check runtime availability
        if not await self._runtime.is_available():
            logger.error(
                "%s runtime is not available. Is it installed and running?",
                self._runtime.runtime_name,
            )
            return False

        # Discover installed models
        await self.discover_models()

        if not self._available_models:
            logger.warning("No models found. Use '%s pull <model>' to download one.", self._runtime.runtime_name.lower())
            return True  # Runtime works, just no models yet

        # Auto-select a general model
        general_model = self.select_model(TaskType.GENERAL)
        if general_model:
            self._active_model = general_model
            logger.info("Auto-selected general model: %s", general_model)

        return True

    async def discover_models(self) -> list[ModelInfo]:
        """Discover all locally installed models.

        Returns:
            List of installed ModelInfo objects.
        """
        self._available_models = await self._runtime.list_models()
        logger.info("Discovered %d installed model(s)", len(self._available_models))
        return self._available_models

    def select_model(self, task_type: TaskType = TaskType.GENERAL) -> str | None:
        """Select the best model for a task type.

        Uses the cached selection if available, otherwise runs selection.

        Args:
            task_type: Type of task to select a model for.

        Returns:
            Model name or None if no compatible model found.
        """
        # Check cache
        if task_type.value in self._model_cache:
            cached = self._model_cache[task_type.value]
            # Verify it's still available
            available_names = {m.name for m in self._available_models}
            if cached in available_names:
                return cached

        # Run selection
        selected = self._selector.select_model(
            task_type=task_type,
            profile=self._profile,
            available_models=self._available_models,
        )

        if selected:
            self._model_cache[task_type.value] = selected
            logger.info("Selected model for %s: %s", task_type.value, selected)

        return selected

    async def set_active_model(self, model_name: str) -> bool:
        """Manually set the active model.

        Args:
            model_name: Model to activate.

        Returns:
            True if model exists and was activated.
        """
        available_names = {m.name for m in self._available_models}
        if model_name not in available_names:
            logger.error("Model '%s' is not installed", model_name)
            return False

        self._active_model = model_name
        logger.info("Active model set to: %s", model_name)
        return True

    async def chat(
        self,
        messages: list[ChatMessage],
        *,
        model: str | None = None,
        task_type: TaskType = TaskType.GENERAL,
        stream: bool = False,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> str:
        """Send a chat request using the appropriate model.

        Args:
            messages: Chat messages.
            model: Specific model to use (overrides auto-selection).
            task_type: Task type for auto model selection.
            stream: Whether to stream (non-streaming for simple returns).
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.

        Returns:
            The model's response text.
        """
        model_name = model or self._resolve_model(task_type)
        if not model_name:
            raise ModelNotAvailableError("No model available for this task")

        self._active_model = model_name
        return await self._runtime.chat(
            model=model_name,
            messages=messages,
            stream=stream,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    async def chat_stream(
        self,
        messages: list[ChatMessage],
        *,
        model: str | None = None,
        task_type: TaskType = TaskType.GENERAL,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ):
        """Stream chat tokens using the appropriate model.

        Args:
            messages: Chat messages.
            model: Specific model to use.
            task_type: Task type for auto selection.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.

        Yields:
            Response tokens/chunks.
        """
        model_name = model or self._resolve_model(task_type)
        if not model_name:
            raise ModelNotAvailableError("No model available for this task")

        self._active_model = model_name
        async for token in self._runtime.chat_stream(
            model=model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        ):
            yield token

    async def unload_current_model(self) -> bool:
        """Unload the currently active model to free resources."""
        if self._active_model:
            success = await self._runtime.unload_model(self._active_model)
            if success:
                logger.info("Unloaded model: %s", self._active_model)
                self._active_model = None
            return success
        return True

    def get_recommended_models(self) -> dict[str, str]:
        """Get recommended models for all task types based on hardware."""
        return self._selector.get_recommended_models(self._profile)

    def get_status(self) -> dict[str, Any]:
        """Get current model manager status."""
        return {
            "runtime": self._runtime.runtime_name,
            "active_model": self._active_model,
            "available_models": len(self._available_models),
            "tier": self._profile.tier.label,
            "max_model_size_gb": self._profile.max_model_size_gb,
            "model_cache": dict(self._model_cache),
        }

    def _resolve_model(self, task_type: TaskType) -> str | None:
        """Resolve which model to use for a task type."""
        # Use active model if set and no specific task model
        if self._active_model:
            return self._active_model

        # Auto-select based on task type
        return self.select_model(task_type)


class ModelNotAvailableError(Exception):
    """Raised when no suitable model is available for a task."""

    pass
