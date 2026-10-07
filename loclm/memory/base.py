"""Base interfaces and data structures for LocLM Local Memory (V5)."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any
from pydantic import BaseModel, Field


class MemoryEntry(BaseModel):
    """Represents a stored item in local memory."""

    entry_id: str = Field(description="Unique identifier for the memory entry")
    content: str = Field(description="Text content of the memory entry")
    category: str = Field(default="general", description="Category: project, tool_history, fact, conversation")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Arbitrary key-value metadata")
    timestamp: float = Field(default_factory=time.time, description="Unix timestamp of entry creation")


class MemorySearchResult(BaseModel):
    """Result returned from a memory search query."""

    entry: MemoryEntry
    score: float = Field(description="Relevance or similarity score (0.0 to 1.0)")


class BaseMemoryStore(ABC):
    """Abstract interface for local memory storage backends."""

    @abstractmethod
    async def add(self, content: str, category: str = "general", metadata: dict[str, Any] | None = None) -> MemoryEntry:
        """Add a new memory entry."""
        pass

    @abstractmethod
    async def search(self, query: str, limit: int = 5, category: str | None = None) -> list[MemorySearchResult]:
        """Search memory entries by query relevance."""
        pass

    @abstractmethod
    async def clear(self, category: str | None = None) -> None:
        """Clear memory entries."""
        pass

    @abstractmethod
    async def get_all(self, limit: int = 100) -> list[MemoryEntry]:
        """Retrieve stored memory entries."""
        pass
