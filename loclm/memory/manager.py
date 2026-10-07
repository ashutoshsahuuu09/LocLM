"""Memory Manager for LocLM V5 Architecture.

Coordinates memory indexing, context retrieval for agents,
and saving conversation and tool reflections into persistent memory.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from loclm.memory.base import BaseMemoryStore, MemoryEntry, MemorySearchResult
from loclm.memory.local_store import SQLiteMemoryStore

logger = logging.getLogger(__name__)


class MemoryManager:
    """Manages short-term and long-term memory operations."""

    def __init__(self, store: BaseMemoryStore | None = None) -> None:
        self.store = store or SQLiteMemoryStore()

    async def add_fact(self, content: str, metadata: dict[str, Any] | None = None) -> MemoryEntry:
        """Add a persistent user fact or codebase context rule."""
        return await self.store.add(content=content, category="fact", metadata=metadata)

    async def record_task_run(self, task: str, result: str, metadata: dict[str, Any] | None = None) -> MemoryEntry:
        """Record completed task execution and outcome into memory."""
        combined_content = f"Task: {task}\nResult: {result}"
        meta = metadata or {}
        meta["task"] = task
        return await self.store.add(content=combined_content, category="task_history", metadata=meta)

    async def get_relevant_context(self, query: str, limit: int = 3) -> str:
        """Fetch and format relevant memory context snippets for prompt enrichment.

        Args:
            query: The user prompt or task string.
            limit: Maximum memory snippets to retrieve.

        Returns:
            Formatted context string for inclusion in agent prompt.
        """
        results = await self.store.search(query=query, limit=limit)
        if not results:
            return ""

        context_lines = ["[Relevant Offline Memory Context]:"]
        for res in results:
            context_lines.append(f"- (Score: {res.score:.2f}) {res.entry.content}")

        return "\n".join(context_lines)
