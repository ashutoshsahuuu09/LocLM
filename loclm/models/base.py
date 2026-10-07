"""Abstract base classes for model runtimes.

Agents depend on these abstractions, NOT on specific runtimes.
This allows swapping Ollama for llama.cpp or any other local runtime.
"""

from __future__ import annotations

import abc
from collections.abc import AsyncIterator
from enum import StrEnum

from pydantic import BaseModel, Field


class TaskType(StrEnum):
    """Types of tasks that require different model specializations."""

    GENERAL = "general"
    CODING = "coding"
    REASONING = "reasoning"
    EMBEDDING = "embedding"
    VISION = "vision"
    ROUTER = "router"


class ModelInfo(BaseModel):
    """Information about an available model."""

    name: str = Field(description="Model name (e.g., 'qwen2.5:7b')")
    size_gb: float = Field(default=0.0, description="Model size on disk in GB")
    family: str = Field(default="unknown", description="Model family (qwen, llama, etc.)")
    parameter_size: str = Field(default="", description="Parameter count (e.g., '7B')")
    quantization: str = Field(default="", description="Quantization level (e.g., 'Q4_K_M')")
    format: str = Field(default="", description="Model format (gguf, etc.)")
    digest: str = Field(default="", description="Model digest/hash")
    modified_at: str = Field(default="", description="Last modified timestamp")

    @property
    def display_name(self) -> str:
        """Human-readable display name."""
        parts = [self.name]
        if self.quantization:
            parts.append(f"({self.quantization})")
        if self.size_gb > 0:
            parts.append(f"[{self.size_gb:.1f}GB]")
        return " ".join(parts)


class ChatMessage(BaseModel):
    """A single message in a chat conversation."""

    role: str = Field(description="Message role: system, user, assistant")
    content: str = Field(description="Message content")


class ModelRuntime(abc.ABC):
    """Abstract base class for local model inference runtimes.

    All concrete runtimes (Ollama, llama.cpp, etc.) must implement this.
    Agents interact with models ONLY through this interface.
    """

    @abc.abstractmethod
    async def is_available(self) -> bool:
        """Check if the runtime is installed and responding."""
        ...

    @abc.abstractmethod
    async def list_models(self) -> list[ModelInfo]:
        """List all locally installed models."""
        ...

    @abc.abstractmethod
    async def model_info(self, model_name: str) -> ModelInfo | None:
        """Get detailed info about a specific model."""
        ...

    @abc.abstractmethod
    async def pull_model(self, model_name: str) -> bool:
        """Download/pull a model to local storage."""
        ...

    @abc.abstractmethod
    async def chat(
        self,
        model: str,
        messages: list[ChatMessage],
        *,
        stream: bool = True,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> str:
        """Send a chat completion request and return the full response.

        Args:
            model: Model name to use.
            messages: List of chat messages.
            stream: Whether to stream internally (for progress display).
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.

        Returns:
            The assistant's response text.
        """
        ...

    @abc.abstractmethod
    async def chat_stream(
        self,
        model: str,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        """Stream chat completion tokens.

        Args:
            model: Model name to use.
            messages: List of chat messages.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.

        Yields:
            Individual response tokens/chunks.
        """
        ...

    @abc.abstractmethod
    async def generate(
        self,
        model: str,
        prompt: str,
        *,
        stream: bool = False,
        temperature: float = 0.7,
    ) -> str:
        """Simple text generation without chat format.

        Args:
            model: Model name to use.
            prompt: Text prompt.
            stream: Whether to stream.
            temperature: Sampling temperature.

        Returns:
            Generated text.
        """
        ...

    @abc.abstractmethod
    async def is_model_loaded(self, model_name: str) -> bool:
        """Check if a model is currently loaded in memory."""
        ...

    @abc.abstractmethod
    async def unload_model(self, model_name: str) -> bool:
        """Unload a model from memory to free resources."""
        ...

    @property
    @abc.abstractmethod
    def runtime_name(self) -> str:
        """Human-readable name of this runtime."""
        ...
